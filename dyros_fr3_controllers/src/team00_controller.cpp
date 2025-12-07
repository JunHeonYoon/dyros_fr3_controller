#include "dyros_fr3_controllers/team00_controller.h"
#include <dyros_fr3_controllers/robot_utils.hpp>

namespace dyros_fr3_controllers 
{
    // ========================================================================
    // ============================ Core Functions ============================
    // ========================================================================
    Team00Controller::~Team00Controller()
    {
        {
            std::lock_guard<std::mutex> lock(compute_cv_mutex_);
            stop_compute_thread_ = true;
            compute_requested_ = false;
        }
        compute_cv_.notify_all();
        compute_done_cv_.notify_all();
        if (compute_thread_.joinable())
        {
            compute_thread_.join();
        }
    }

    CallbackReturn Team00Controller::on_init() 
    {
        try 
        {
            auto_declare<std::string>("arm_id", "");

            const std::string pkg_share = ament_index_cpp::get_package_share_directory("dyros_fr3_controllers");
            urdf_path = pkg_share + "/urdf/fr3_franka_hand.urdf";
            std::ifstream urdf_file(urdf_path);
            if (!urdf_file.good()) 
            {
                LOGW(get_node(), "URDF not found at path: %s (Pinocchio will be disabled)", urdf_path.c_str());
                use_pinocchio_ = false;
            }

            if(use_pinocchio_)
            {
                pinocchio::urdf::buildModel(urdf_path, model_);
                data_ = pinocchio::Data(model_);
            }

            q_.setZero();
            qdot_.setZero();
            tau_.setZero();

            q_desired_.setZero();
            qdot_desired_.setZero();

            tau_desired_.setZero();

            q_init_.setZero();
            qdot_init_.setZero();

            x_.setIdentity();
            xdot_.setZero();
            J_.setZero();
            x_init_.setIdentity();
            xdot_init_.setZero();

            x2_.setIdentity();
            x2dot_.setZero();
            J2_.setZero();
            x2_init_.setIdentity();
            x2dot_init_.setZero();

            M_.setIdentity();
            M_inv_.setIdentity();
            c_.setZero();
            // g_.setZero();

            logging_file_.open("logging.txt");

            control_mode_sub_ = get_node()->create_subscription<std_msgs::msg::Int32>(
                "team00_controller/control_mode",
                rclcpp::QoS(10),
                std::bind(&Team00Controller::keyMapping, this, std::placeholders::_1)
            );

            if (!compute_thread_.joinable())
            {
                std::lock_guard<std::mutex> lock(compute_cv_mutex_);
                stop_compute_thread_ = false;
                compute_requested_ = false;
                compute_completed_ = false;
                compute_thread_ = std::thread(&Team00Controller::computeWorkerLoop, this);
            }

        } 
        catch (const std::exception& e) 
        {
            LOGE(get_node(), "Exception during initialization: %s", e.what());
            return CallbackReturn::ERROR;
        }
        return CallbackReturn::SUCCESS;
    }

    controller_interface::InterfaceConfiguration Team00Controller::command_interface_configuration() const 
    {
        controller_interface::InterfaceConfiguration command_interface_config;
        command_interface_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
        for (int i = 1; i <= num_joints; ++i) 
        {
            command_interface_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
        }
        return command_interface_config;
    }

    controller_interface::InterfaceConfiguration Team00Controller::state_interface_configuration() const 
    {
        controller_interface::InterfaceConfiguration state_interfaces_config;
        state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
        for (int i = 1; i <= num_joints; ++i) 
        {
            state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
        }
        for (int i = 1; i <= num_joints; ++i) 
        {
            state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
        }
        for (int i = 1; i <= num_joints; ++i) 
        {
            state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
        }
        for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) 
        {
            state_interfaces_config.names.push_back(franka_robot_model_name);
        }
        state_interfaces_config.names.push_back(arm_id_ + "/robot_time");
        return state_interfaces_config;
    }

    CallbackReturn Team00Controller::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) 
    {
        if (get_node()->get_parameter("arm_id", arm_id_)) 
        {
            arm_id_ = get_node()->get_parameter("arm_id").as_string();
        }
        else
        {
            LOGW(get_node(), "Parameter 'arm_id' not set — using defaults fr3");
            arm_id_ = "fr3";
        }

        franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + "robot_model",
        arm_id_ + "/" + "robot_state"));

        return CallbackReturn::SUCCESS;
    }

    CallbackReturn Team00Controller::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) 
    {
        initialization_flag_ = true;

        franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

        updateJointStates();
        updateRobotData();

        {
            std::lock_guard<std::mutex> lk(robot_data_mutex_);
            tau_desired_ = c_;
        }

        LOGI(get_node(), "Controller activated (arm_id: %s, dt: %.3f ms)", arm_id_.c_str(), dt_ * 1000.0);
        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn Team00Controller::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) 
    {
        franka_robot_model_->release_interfaces();
        return CallbackReturn::SUCCESS;
    }

    controller_interface::return_type Team00Controller::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period)
    {
        SuhanBenchmark bench;

        updateJointStates();
        updateRobotData();

        {
            std::lock_guard<std::mutex> lk(robot_data_mutex_);
            play_time_ = state_interfaces_.back().get_value();
        }

        const double spent_ms = bench.elapsed() * 1000.;
        double budget_ms = (dt_*1000.) - spent_ms - 0.05; // 0.05 ms for command to robot
        if (budget_ms < 0.0) 
        {
            budget_ms = 0.0;
            LOGW(get_node(), "State update exceeded 1.0 ms (%.3f ms)", spent_ms);
        }

        constexpr double kWaitGuardMs = 0.2; // leave headroom for remaining work and comms
        const bool skip_guard = relax_wait_guard_.load(std::memory_order_acquire);
        if (!skip_guard && budget_ms > kWaitGuardMs)
        {
            budget_ms -= kWaitGuardMs;
        }
        else if (!skip_guard)
        {
            budget_ms = 0.0;
        }

        Vector7d last_command;
        {
            std::lock_guard<std::mutex> lk(calculation_mutex_);
            last_command = tau_desired_;
        }

        bool used_new_solution = false;

        auto request_compute_and_wait = [this](double wait_ms, bool clear_relax_on_success) -> bool
            {
                {
                    std::lock_guard<std::mutex> lock(compute_cv_mutex_);
                    compute_requested_ = true;
                    compute_completed_ = false;
                }
                compute_cv_.notify_one();

                const double clamped_wait = std::max(0.0, std::min(wait_ms, 0.2));
                const auto wait_dur = std::chrono::duration<double, std::milli>(clamped_wait);
                std::unique_lock<std::mutex> wait_lock(compute_cv_mutex_);
                if (compute_done_cv_.wait_for(wait_lock, wait_dur, [this]() { return compute_completed_; }))
                {
                    if (clear_relax_on_success)
                    {
                        relax_wait_guard_.store(false, std::memory_order_release);
                    }
                    return true;
                }
                return false;
            };
        if (skip_guard)
        {
            if (!compute_inflight_.exchange(true, std::memory_order_acq_rel))
            {
                try
                {
                    compute();
                    used_new_solution = true;
                    relax_wait_guard_.store(false, std::memory_order_release);
                }
                catch (const std::exception& e)
                {
                    LOGE(get_node(), "Exception in compute(): %s", e.what());
                }
                catch (...)
                {
                    LOGE(get_node(), "Unknown exception in compute()");
                }
                compute_inflight_.store(false, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lock(compute_cv_mutex_);
                    compute_completed_ = true;
                    compute_requested_ = false;
                }
                compute_done_cv_.notify_all();
            }
        }
        else if (!compute_inflight_.exchange(true, std::memory_order_acq_rel)) 
        {
            if (request_compute_and_wait(budget_ms, false))
            {
                used_new_solution = true;
            }
        } 

        Vector7d command;
        if (used_new_solution) 
        {
            std::lock_guard<std::mutex> lk(calculation_mutex_);
            command = tau_desired_;
            relax_wait_guard_.store(false, std::memory_order_release);
        } 
        else 
        {
            command = last_command;
        }

        for (int i = 0; i < num_joints; ++i) 
        {
            command_interfaces_[i].set_value(command[i]);
        }

        printState();

        return controller_interface::return_type::OK;
    }

    void Team00Controller::keyMapping(const std_msgs::msg::Int32& msg)
    {
        LOGI(get_node(), "Mode input received: %d", msg.data);
        switch (msg.data)
        {
            // Implement with user input
            case 1:
                setMode(joint_ctrl_home);
                break;

            // --------------------------------------------------------------------------------------
            // TODO 3: Add your keyboard mapping here (using the control modes from TODO 1)
            // Caution: not using string, but int on case!!!!
            // Example:
            case 2:
                setMode(joint_ctrl_home_1);
                break;
            case 3:
                setMode(hw8_1);
                break;
            case 4:
                setMode(hw8_2);
                break;
            case 5:
                setMode(hw8_3);
                break;
            case 6:
                setMode(joint_ctrl_home_2);
                break;
            case 7:
                setMode(hw8_4);
                break;
            case 8:
                setMode(hw8_5);
                break;
            case 9:
                setMode(hw8_6);
                break;
            // --------------------------------------------------------------------------------------

            default:
                LOGW(get_node(), "Unknown mode value: %d (ignored)", msg.data);
                break;
        }
    }

    void Team00Controller::compute()
    {
        std::scoped_lock(robot_data_mutex_, calculation_mutex_);
        if (initialization_flag_ || is_mode_changed_) 
        {            
            initialization_flag_ = false;
            is_mode_changed_ = false;
            control_start_time_ = play_time_;
            q_init_ = q_;
            qdot_init_ = qdot_;
            x_init_ = x_;
            x2_init_ = x2_;
            xdot_init_ = xdot_;
            x2dot_init_ = x2dot_;
            q_desired_ = q_;
            qdot_desired_ = qdot_;
        }

        switch (control_mode_)
        {
            case joint_ctrl_home:
            {
                Vector7d target_position;
                target_position << 0.0, 0.0, 0.0, -M_PI/2., 0.0, M_PI/2., M_PI / 4.;
                moveJointPositionTorque(target_position, 3.0);
                break;
            }
            // ----------------------------------------------------------------------------
            // TODO 4: Add your control logic here (using the control functions from TODO 5)
            // Example:
            case joint_ctrl_home_1:
            {
                Vector7d target_position;
                target_position << 0.0, 0.0, 0.0, -30*DEG2RAD, 0.0, 90*DEG2RAD, 0.;
                moveJointPositionTorque(target_position, 3.0);
                break;
            }
            case joint_ctrl_home_2:
            {
                Vector7d target_position;
                target_position << 0.0, 0.0, 0.0, -M_PI/2., 0.0, M_PI/2., M_PI / 4.;
                moveJointPositionTorque(target_position, 3.0);
                break;
            }
            case hw8_1:
            {
                HW8_1();
                break;
            }
            case hw8_2:
            {
                Vector7d q_target,qdot_target;
                q_target << 0.0, 0.0, 0.0, -60*DEG2RAD, 0.0, 90*DEG2RAD, 0.;
                qdot_target.setZero();
                HW8_2(q_target, qdot_target, 2.0);
                break;
            }
            case hw8_3:
            {
                Vector7d q_target,qdot_target;
                q_target << 0.0, 0.0, 0.0, -60*DEG2RAD, 0.0, 90*DEG2RAD, 0.;
                qdot_target.setZero();
                HW8_3(q_target, qdot_target, 2.0);
                break;
            }
            case hw8_4:
            {
                Matrix4d x_target;
                x_target.setIdentity();
                x_target = x_init_;
                x_target(1, 3) += 0.1;
                HW8_4(x_target, 2.0);
                break;
            }
            case hw8_5:
            {
                Matrix4d x_target;
                x_target.setIdentity();
                x_target = x_init_;
                x_target(1, 3) += 0.1;
                HW8_5(x_target, 2.0);
                break;
            }
            case hw8_6:
            {
                Matrix4d x_target;
                x_target.setIdentity();
                x_target = x_init_;
                x_target.block(0,3,3,1) << 0.3, -0.012, 0.52;
                HW8_6(x_target, 2.0);
                break;
            }
            
            default:
            {
                tau_desired_ = c_;
                break;
            }
            // ----------------------------------------------------------------------------
        }
    }

    // ============================================================================
    // =============================== User functions ==============================
    // =============================================================================
    void Team00Controller::moveJointPositionTorque(const Vector7d& target_position, double duration)
    {
        Vector7d kp_diag, kv_diag;
        Vector7d q_cubic, qd_cubic;

        kp_diag << 600.0,600.0,600.0,600.0,250.0,150.0,50.0;
        kv_diag << 30.0,30.0,30.0,30.0,10.0,10.0,5.0;


        for (int i = 0; i < 7; i++)
        {
            qd_cubic(i) = DyrosMath::cubicDot(play_time_, 
                                              control_start_time_,
                                              control_start_time_ + duration, 
                                              q_init_(i), 
                                              target_position(i), 
                                              0, 
                                              0);
            q_cubic(i) = DyrosMath::cubic(play_time_,
                                          control_start_time_,
                                          control_start_time_ + duration, 
                                          q_init_(i), 
                                          target_position(i), 
                                          0, 
                                          0);
        }


        tau_desired_ = (kp_diag.asDiagonal()*(q_cubic - q_) + kv_diag.asDiagonal()*(qd_cubic - qdot_)) + c_;
    }

    // -------------------------------------
    // TODO 5: Add your control functions here
    // Example:
    void Team00Controller::HW8_1()
    {
        tau_desired_ = c_;
    }

    void Team00Controller::HW8_2(const Vector7d& q_target, const Vector7d& qdot_target, const double duration)
    {
        Vector7d Kp_diag, Kv_diag;
        Kp_diag << 600.0,600.0,600.0,600.0,250.0,150.0,50.0;
        Kv_diag << 30.0,30.0,30.0,30.0,10.0,10.0,5.0;

        q_desired_ = DyrosMath::cubicVector<7>(play_time_,
                                               control_start_time_,
                                               control_start_time_+duration,
                                               q_init_,
                                               q_target,
                                               qdot_init_,
                                               qdot_target);

        qdot_desired_ = DyrosMath::cubicDotVector<7>(play_time_,
                                                     control_start_time_,
                                                     control_start_time_+duration,
                                                     q_init_,
                                                     q_target,
                                                     qdot_init_,
                                                     qdot_target);

        tau_desired_ = Kp_diag.asDiagonal() * (q_desired_ - q_) + Kv_diag.asDiagonal() * (qdot_desired_ - qdot_) + c_;
    }

    void Team00Controller::HW8_3(const Vector7d& q_target, const Vector7d& qdot_target, const double duration)
    {
        Vector7d Kp_diag, Kv_diag;
        Kp_diag << 600.0,600.0,600.0,600.0,250.0,150.0,50.0;
        Kv_diag << 30.0,30.0,30.0,30.0,10.0,10.0,5.0;

        q_desired_ = DyrosMath::cubicVector<7>(play_time_,
                                               control_start_time_,
                                               control_start_time_+duration,
                                               q_init_,
                                               q_target,
                                               qdot_init_,
                                               qdot_target);

        qdot_desired_ = DyrosMath::cubicDotVector<7>(play_time_,
                                                     control_start_time_,
                                                     control_start_time_+duration,
                                                     q_init_,
                                                     q_target,
                                                     qdot_init_,
                                                     qdot_target);

        tau_desired_ = M_ * (Kp_diag.asDiagonal() * (q_desired_ - q_) + Kv_diag.asDiagonal() * (qdot_desired_ - qdot_)) + c_;
    }
    
    void Team00Controller::HW8_4(const Matrix4d x_target, const double duration)
    {
        Matrix4d x_desired;
        x_desired.setIdentity();
        x_desired.block(0,3,3,1) = DyrosMath::cubicVector<3>(play_time_,
                                                            control_start_time_,
                                                            control_start_time_+duration,
                                                            x_init_.block(0,3,3,1),
                                                            x_target.block(0,3,3,1),
                                                            VectorXd::Zero(3),
                                                            VectorXd::Zero(3));
        x_desired.block(0,0,3,3) = x_target.block(0,0,3,3);

        Vector6d xdot_desired;
        xdot_desired.head(3) = DyrosMath::cubicDotVector<3>(play_time_,
                                                            control_start_time_,
                                                            control_start_time_+duration,
                                                            x_init_.block(0,3,3,1),
                                                            x_target.block(0,3,3,1),
                                                            VectorXd::Zero(3),
                                                            VectorXd::Zero(3));
        xdot_desired.tail(3).setZero();

        Vector6d x_error, xdot_error;
        x_error.head(3) = x_desired.block(0,3,3,1) - x_.block(0,3,3,1);
        x_error.tail(3) = -DyrosMath::getPhi(x_.block(0,0,3,3), x_desired.block(0,0,3,3));
        xdot_error = xdot_desired - xdot_;
        
        Vector6d Kp_diag, Kv_diag;
        Kp_diag << 400, 400, 400, 400, 400, 400;
        Kv_diag << 40, 40, 40, 40, 40, 40;

        Vector6d Fstar = Kp_diag.asDiagonal() * x_error + Kv_diag.asDiagonal() * xdot_error;
        Matrix6d M_task = (J_ * M_inv_ * J_.transpose()).inverse();
        
        tau_desired_ = J_.transpose() * M_task * Fstar + c_;
    }

    void Team00Controller::HW8_5(const Matrix4d x_target, const double duration)
    {
        Matrix4d x_desired;
        x_desired.setIdentity();
        x_desired.block(0,3,3,1) = DyrosMath::cubicVector<3>(play_time_,
                                                            control_start_time_,
                                                            control_start_time_+duration,
                                                            x_init_.block(0,3,3,1),
                                                            x_target.block(0,3,3,1),
                                                            VectorXd::Zero(3),
                                                            VectorXd::Zero(3));
        x_desired.block(0,0,3,3) = x_target.block(0,0,3,3);

        Vector6d xdot_desired;
        xdot_desired.head(3) = DyrosMath::cubicDotVector<3>(play_time_,
                                                            control_start_time_,
                                                            control_start_time_+duration,
                                                            x_init_.block(0,3,3,1),
                                                            x_target.block(0,3,3,1),
                                                            VectorXd::Zero(3),
                                                            VectorXd::Zero(3));
        xdot_desired.tail(3).setZero();

        Vector6d x_error, xdot_error;
        x_error.head(3) = x_desired.block(0,3,3,1) - x_.block(0,3,3,1);
        x_error.tail(3) = -DyrosMath::getPhi(x_.block(0,0,3,3), x_desired.block(0,0,3,3));
        xdot_error = xdot_desired - xdot_;
        
        Vector6d Kp_diag, Kv_diag;
        Kp_diag << 400, 400, 400, 400, 400, 400;
        Kv_diag << 40, 40, 40, 40, 40, 40;

        Vector6d Fstar = Kp_diag.asDiagonal() * x_error + Kv_diag.asDiagonal() * xdot_error;

        Vector7d Kp_joint_diag, Kv_joint_diag;
        Kp_joint_diag << 600.0,600.0,600.0,600.0,250.0,150.0,50.0;
        Kv_joint_diag << 30.0,30.0,30.0,30.0,10.0,10.0,5.0;

        Vector7d tau_null = M_ * (Kp_joint_diag.asDiagonal() * (q_init_ - q_) + Kv_joint_diag.asDiagonal() * (-qdot_));

        Matrix6d M_task = (J_ * M_inv_ * J_.transpose()).inverse();
        Matrix<double, 6, 7> J_T_pinv = M_task * J_ * M_inv_;
        
        tau_desired_ = J_.transpose() * M_task * Fstar + (Matrix7d::Identity() - J_.transpose() * J_T_pinv) * tau_null + c_;
    }

    void Team00Controller::HW8_6(const Matrix4d x_target, const double duration)
    {
        Matrix4d x_desired;
        x_desired.setIdentity();
        x_desired.block(0,3,3,1) = DyrosMath::cubicVector<3>(play_time_,
                                                            control_start_time_,
                                                            control_start_time_+duration,
                                                            x_init_.block(0,3,3,1),
                                                            x_target.block(0,3,3,1),
                                                            VectorXd::Zero(3),
                                                            VectorXd::Zero(3));
        x_desired.block(0,0,3,3) = x_target.block(0,0,3,3);

        Vector6d Kp_diag, Kv_diag, Kp_Kv_diag;
        Kp_diag << 400, 400, 400, 400, 400, 400;
        Kv_diag << 40, 40, 40, 40, 40, 40;
        Kp_Kv_diag = Kp_diag.array() / Kv_diag.array();

        Vector6d x_error;
        x_error.head(3) = x_desired.block(0,3,3,1) - x_.block(0,3,3,1);
        x_error.tail(3) = -DyrosMath::getPhi(x_.block(0,0,3,3), x_desired.block(0,0,3,3));

        double xdot_max = 0.3;

        Vector6d xdot_desired;
        if((Kp_Kv_diag.head(3).asDiagonal() * x_error.head(3)).norm() < xdot_max) xdot_desired.head(3) = Kp_Kv_diag.head(3).asDiagonal() * x_error.head(3);
        else xdot_desired.head(3) =x_error.head(3).normalized() * xdot_max;
        xdot_desired.tail(3).setZero();

        Vector6d Fstar;
        Fstar.head(3) = Kv_diag.head(3).asDiagonal() * (xdot_desired.head(3) - xdot_.head(3));
        Fstar.tail(3) = Kp_diag.tail(3).asDiagonal() * x_error.tail(3) - Kv_diag.tail(3).asDiagonal() * xdot_.tail(3);

        Vector7d Kp_joint_diag, Kv_joint_diag;
        Kp_joint_diag << 600.0,600.0,600.0,600.0,250.0,150.0,50.0;
        Kv_joint_diag << 30.0,30.0,30.0,30.0,10.0,10.0,5.0;

        Vector7d tau_null = M_ * (Kp_joint_diag.asDiagonal() * (q_init_ - q_) + Kv_joint_diag.asDiagonal() * (-qdot_));

        Matrix6d M_task = (J_ * M_inv_ * J_.transpose()).inverse();
        Matrix<double, 6, 7> J_T_pinv = M_task * J_ * M_inv_;
        
        tau_desired_ = J_.transpose() * M_task * Fstar + (Matrix7d::Identity() - J_.transpose() * J_T_pinv) * tau_null +c_;
    }
    // -------------------------------------
    // =============================================================================

    void Team00Controller::setMode(const CTRL_MODE& control_mode)
    {
        std::scoped_lock<std::mutex, std::mutex> lk(robot_data_mutex_, calculation_mutex_);

        control_mode_ = control_mode;
        is_mode_changed_ = true;

        LOGI(get_node(), "Mode changed: %d", static_cast<int>(control_mode_));
    }

    void Team00Controller::printState()
    {
        static int DBG_CNT = 0;
        if (DBG_CNT++ > 1 / (dt_* 50.))
        {
            DBG_CNT = 0;
            // TODO 6: Extend or modify this for debugging your controller
            std::cout << "\n\n------------------------------------------------------------------" << std::endl;
            std::cout << "time     : " << std::fixed << std::setprecision(3) << play_time_ << std::endl;
            std::cout << "q now    :\t";
            std::cout << std::fixed << std::setprecision(3) << q_.transpose() << std::endl;
            std::cout << "x        :\n";
            std::cout << std::fixed << std::setprecision(3) << x_ << std::endl;
            std::cout << "x dot    :\t";
            std::cout << std::fixed << std::setprecision(3) << xdot_.transpose() << std::endl;
            std::cout << "J        :\n";
            std::cout << std::fixed << std::setprecision(3) << J_ << std::endl;
        }
    }

    Matrix4d Team00Controller::getLinkPose(const VectorXd& q, const std::string& link_name)
    {
        if(q.size() != model_.nq)
        {
            LOGE(get_node(), "getEEPose Error: size of q %d is not equal to model.nq size: %d", q.size(), model_.nq);
            return Matrix4d::Identity();
        }
        pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
        if (link_index == static_cast<pinocchio::FrameIndex>(-1))  
        {
            LOGE(get_node(), "Error: Link name %s not found in URDF.", ee_name_.c_str());
            return Matrix4d::Identity();
        }

        pinocchio::Data data_tmp(model_);
        pinocchio::framesForwardKinematics(model_, data_tmp, q);
        return data_tmp.oMf[link_index].toHomogeneousMatrix();
    }

    MatrixXd Team00Controller::getLinkJac(const VectorXd& q, const std::string& link_name)
    {
        if(q.size() != model_.nq)
        {
            LOGE(get_node(), "getEEJac Error: size of q %d is not equal to model.nq size: %d", q.size(), model_.nq);
            return MatrixXd::Zero(6, model_.nv);
        }
        pinocchio::FrameIndex link_index = model_.getFrameId(link_name);
        if (link_index == static_cast<pinocchio::FrameIndex>(-1))  
        {
            LOGE(get_node(), "Error: Link name %s not found in URDF.", ee_name_.c_str());
            return MatrixXd::Zero(6, model_.nv);
        }

        MatrixXd J;
        J.setZero(6, model_.nv);
        pinocchio::Data data_tmp(model_);
        pinocchio::computeJointJacobians(model_, data_tmp, q);
        pinocchio::getFrameJacobian(model_, data_tmp, link_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);

        return J;
    }

    Matrix4d Team00Controller::getEEPose(const VectorXd& q)
    {
        if(q.size() != model_.nq)
        {
            LOGE(get_node(), "getEEPose Error: size of q %d is not equal to model.nq size: %d", q.size(), model_.nq);
            return Matrix4d::Identity();
        }
        pinocchio::FrameIndex ee_index = model_.getFrameId(ee_name_);
        if (ee_index == static_cast<pinocchio::FrameIndex>(-1))  
        {
            LOGE(get_node(), "Error: Link name %s not found in URDF.", ee_name_.c_str());
            return Matrix4d::Identity();
        }

        pinocchio::Data data_tmp(model_);
        pinocchio::framesForwardKinematics(model_, data_tmp, q);
        return data_tmp.oMf[ee_index].toHomogeneousMatrix();
    }

    MatrixXd Team00Controller::getEEJac(const VectorXd& q)
    {
        if(q.size() != model_.nq)
        {
            LOGE(get_node(), "getEEJac Error: size of q %d is not equal to model.nq size: %d", q.size(), model_.nq);
            return MatrixXd::Zero(6, model_.nv);
        }
        pinocchio::FrameIndex ee_index = model_.getFrameId(ee_name_);
        if (ee_index == static_cast<pinocchio::FrameIndex>(-1))  
        {
            LOGE(get_node(), "Error: Link name %s not found in URDF.", ee_name_.c_str());
            return MatrixXd::Zero(6, model_.nv);
        }

        MatrixXd J;
        J.setZero(6, model_.nv);
        pinocchio::Data data_tmp(model_);
        pinocchio::computeJointJacobians(model_, data_tmp, q);
        pinocchio::getFrameJacobian(model_, data_tmp, ee_index, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);

        return J;
    }

    MatrixXd Team00Controller::getMassMatrix(const VectorXd& q)
    {
        if(q.size() != model_.nq)
        {
            LOGE(get_node(), "getMassMatrix Error: size of q %d is not equal to model.nq size: %d", q.size(), model_.nq);
            return MatrixXd::Zero(model_.nq, model_.nq);
        }
        pinocchio::Data data_tmp(model_);
        pinocchio::crba(model_, data_tmp, q);

        return data_tmp.M.selfadjointView<Upper>();  // Only upper triangular part of M_ is computed by pinocchio::crba
    }

    // VectorXd Team00Controller::getGravityVector(const VectorXd& q)
    // {
    //     if(q.size() != model_.nq)
    //     {
    //         std::cerr << "getGravityVector Error: size of q " << q.size() << " is not equal to model.nq size: " << model_.nq << std::endl;
    //         return VectorXd::Zero(model_.nq);
    //     }
    //     pinocchio::Data data_tmp(model_);
    //     pinocchio::computeGeneralizedGravity(model_, data_tmp, q);

    //     return data_tmp.g;
    // }

    void Team00Controller::updateJointStates() 
    {
        std::lock_guard<std::mutex> lk(robot_data_mutex_);
        for (int i = 0; i < num_joints; ++i) 
        {
            const auto& position_interface = state_interfaces_.at(i);
            const auto& velocity_interface = state_interfaces_.at(num_joints + i);
            const auto& effort_interface = state_interfaces_.at(2 * num_joints + i);
            q_[i] = position_interface.get_value();
            qdot_[i] = velocity_interface.get_value();
            tau_[i] = effort_interface.get_value();
        }
    }

    void Team00Controller::updateRobotData()
    {
        std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
        std::array<double, 7> coriolis = franka_robot_model_->getCoriolisForceVector();
        // std::array<double, 7> gravity = franka_robot_model_->getGravityForceVector();
        std::array<double, 16> pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
        std::array<double, 16> pose2 = franka_robot_model_->getPoseMatrix(franka::Frame::kJoint4);
        std::array<double, 42> endeffector_jacobian_wrt_base = franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
        std::array<double, 42> endeffector_jacobian_wrt_base2 = franka_robot_model_->getZeroJacobian(franka::Frame::kJoint4);
        {
            std::lock_guard<std::mutex> lock(robot_data_mutex_);
            M_ = Map<const Matrix<double, 7, 7, RowMajor>>(mass.data());
            c_ = Map<const Matrix<double, 7, 1>>(coriolis.data());
            // g_ = Map<const Matrix<double, 7, 1>>(gravity.data());

            x_ = Map<const Matrix4d>(pose.data());
            x2_ = Map<const Matrix4d>(pose2.data());

            Map<const Matrix<double,6,7,ColMajor>> J_tmp(endeffector_jacobian_wrt_base.data());
            J_ = J_tmp;
            Map<const Matrix<double,6,7,ColMajor>> J2_tmp(endeffector_jacobian_wrt_base2.data());
            J2_ = J2_tmp;

            M_inv_ = M_.inverse();
            xdot_ = J_ * qdot_;
            x2dot_ = J2_ * qdot_;
        }
    }

    void Team00Controller::computeWorkerLoop()
    {
        std::unique_lock<std::mutex> lock(compute_cv_mutex_);
        while (!stop_compute_thread_)
        {
            compute_cv_.wait(lock, [this]() { return compute_requested_ || stop_compute_thread_; });
            if (stop_compute_thread_) break;
            compute_requested_ = false;
            lock.unlock();
            try
            {
                compute();
            }
            catch (const std::exception& e)
            {
                LOGE(get_node(), "Exception in compute(): %s", e.what());
            }
            catch (...)
            {
                LOGE(get_node(), "Unknown exception in compute()");
            }
            lock.lock();
            compute_completed_ = true;
            compute_inflight_.store(false, std::memory_order_release);
            compute_done_cv_.notify_all();
        }
    }
}  // namespace dyros_fr3_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(dyros_fr3_controllers::Team00Controller,
                       controller_interface::ControllerInterface)

        