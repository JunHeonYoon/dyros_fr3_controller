#pragma once

#include <string>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <array>
#include <cassert>
#include <cmath>
#include <exception>
#include <Eigen/Eigen>
#include <functional> 
#include <future>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/joint/joint-collection.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <controller_interface/controller_interface.hpp>
#include <std_msgs/msg/int32.hpp>
#include "franka_semantic_components/franka_robot_model.hpp"
#include "franka_semantic_components/franka_robot_state.hpp"

#include "math_type_define.h"
#include "suhan_benchmark.h"

namespace ConsoleColor 
{
    inline constexpr const char* RESET = "[0m";
    inline constexpr const char* BLUE  = "[34m"; // Info
    inline constexpr const char* YELLOW= "[33m"; // Warn
    inline constexpr const char* RED   = "[31m"; // Error
}

#define LOGI(node, fmt, ...) RCLCPP_INFO((node)->get_logger(),  (std::string(ConsoleColor::BLUE)   + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)
#define LOGW(node, fmt, ...) RCLCPP_WARN((node)->get_logger(),  (std::string(ConsoleColor::YELLOW) + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)
#define LOGE(node, fmt, ...) RCLCPP_ERROR((node)->get_logger(), (std::string(ConsoleColor::RED)    + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using namespace Eigen;

namespace dyros_fr3_controllers 
{
class Team00Controller : public controller_interface::ControllerInterface 
{
    public:
        ~Team00Controller() override;
        // ========================================================================
        // ============================ Core Functions ============================
        // ========================================================================
        [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration() const override;
        [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration() const override;
        controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;
        CallbackReturn on_init() override;
        CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
        CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
        CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

    private:
        const double dt_{0.001};
            std::ofstream logging_file_;
            // =============================================================================
            // ================================= User data =================================
            // =============================================================================
            // control mode state
            bool is_mode_changed_{false};
            enum CTRL_MODE {
                joint_ctrl_home,

                // -------------------------------------
                // TODO 1: Add your control modes here
                // Example:
                hw8_1,
                // -------------------------------------

                DEFAULT
            };
            CTRL_MODE control_mode_{DEFAULT};

            // time state
            double play_time_{0.0};
            double control_start_time_{0.0};

            // Current Joint space state
            Vector7d q_;            // joint angle (7,1)
            Vector7d qdot_;         // joint velocity (7,1)
            Vector7d tau_;          // joint torque (7,1)

            // Desired Joint space state
            Vector7d q_desired_;    // joint angle (7,1)
            Vector7d qdot_desired_; // joint velocity (7,1)

            // Initial joint space state
            Vector7d q_init_;       // initial joint angle (7,1)
            Vector7d qdot_init_;    // initial joint velocity (7,1)
            
            // Control Input 
            Vector7d tau_desired_;  // desired joint torque (7,1) -> using for torque mode

            // Task space state
            std::vector<std::string> link_names_{"fr3_link0", 
                                                 "fr3_link1", 
                                                 "fr3_link2",
                                                 "fr3_link3",
                                                 "fr3_link4",
                                                 "fr3_link5",
                                                 "fr3_link6",
                                                 "fr3_link7"};
            std::string ee_name_{"fr3_hand_tcp"};
            Matrix4d             x_;            // Homogeneous matrix; pose of EE (4,4)
            Vector6d             xdot_;         // velocity of link4 (6,1); linear + angular
            Matrix<double, 6, 7> J_;            // jacobian of EE (6,7)
            Matrix4d             x_init_;       // Homogeneous matrix; initial pose of EE (4,4)
            Vector6d             xdot_init_;    // velocity of link4 (6,1); linear + angular
            Matrix4d             x2_;           // Homogeneous matrix; pose of link4 (4,4)
            Vector6d             x2dot_;        // velocity of link4 (6,1); linear + angular
            Matrix<double, 6, 7> J2_;           // jacobian of link4 (6,7)
            Matrix4d             x2_init_;      // Homogeneous matrix; initial pose of link4 (4,4)
            Vector6d             x2dot_init_;   // velocity of link4 (6,1); linear + angular

            // Joint space Dynamics
            Matrix7d M_;     // mass matrix (7,7)
            Matrix7d M_inv_; // inverse of mass matrix (7,7)
            // Vector7d g_;     // gravity torques (7,1) // do not use gravity!!!
            Vector7d c_;     // centrifugal and coriolis forces (7,1)

            // =============================================================================
            // =============================== User functions ==============================
            // =============================================================================
            // Control functions
            void moveJointPositionTorque(const Vector7d& target_position, double duration);
            
            // ----------------------------------------------
            // TODO 2: Add your control function here
            // Example:
            void HW8_1();
            // ----------------------------------------------

            // ============================================================================
            // Utils functions
            Matrix4d getLinkPose(const VectorXd& q, const std::string& link_name);
            MatrixXd getLinkJac(const VectorXd& q, const std::string& link_name);
            Matrix4d getEEPose(const VectorXd& q);
            MatrixXd getEEJac(const VectorXd& q);
            MatrixXd getMassMatrix(const VectorXd& q);
            // VectorXd getGravityVector(const VectorXd& q); // do not use gravity!!

            // Core functions
            void keyMapping(const std_msgs::msg::Int32& msg);
            void compute();
            void setMode(const CTRL_MODE& control_mode);
            void printState();
            void updateJointStates();			
            void updateRobotData();
            void computeWorkerLoop();

            // ========================================================================
            // ============================== Parameters ==============================
            // ========================================================================
            std::string arm_id_;
            std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
            const int num_joints = 7;               
            bool initialization_flag_{true};

            // ========================================================================
            // =========================== ROS Subs & Pubs  ===========================
            // ========================================================================
            rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr control_mode_sub_;

            // ========================================================================
            // ============================ Mutex & Thread ============================
            // ========================================================================
            std::mutex robot_data_mutex_;
            std::mutex calculation_mutex_;
            std::atomic<bool> compute_inflight_{false};
            std::atomic<bool> relax_wait_guard_{false};
            std::thread compute_thread_;
            std::mutex compute_cv_mutex_;
            std::condition_variable compute_cv_;
            std::condition_variable compute_done_cv_;
            bool compute_requested_{false};
            bool compute_completed_{false};
            bool stop_compute_thread_{false};


            // ==========================================================================
            // =========================== Pinocchio / Model ============================
            // ==========================================================================
            std::string urdf_path;
            bool use_pinocchio_{true};
            pinocchio::Model model_;
            pinocchio::Data data_;
    };
}  // namespace dyros_fr3_controllers

