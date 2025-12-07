#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

PACKAGE_DIR = "dyros_fr3_controllers"
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_PATH = SCRIPT_DIR / PACKAGE_DIR


def to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def read_file(path: str) -> str:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def append_file(path: str, content: str) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(content)


def update_cmakelists(new_src_rel: str, ctrl_snake: str, controller_class: str) -> None:
# def update_cmakelists(new_src_rel: str) -> None:
    cmake_path = PACKAGE_PATH / "CMakeLists.txt"
    content = read_file(cmake_path)

    # if new_src_rel in content:
    #     return

    # idx = content.find("add_library(")
    # if idx == -1:
    #     write_file(cmake_path, content)
    #     return
    # # Find the closing ')' after the block start
    # close_idx = content.find(")\n", idx)
    # if close_idx == -1:
    #     close_idx = len(content)
    # new_content = content[:close_idx] + f"\n        {new_src_rel}" + content[close_idx:]
    # content = new_content

    # write_file(cmake_path, content)

    if new_src_rel not in content:
        idx = content.find("add_library(")
        if idx != -1:
            close_idx = content.find(")\n", idx)
            if close_idx == -1:
                close_idx = len(content)
            new_content = content[:close_idx] + f"\n        {new_src_rel}" + content[close_idx:]
            content = new_content

    install_marker = f"{ctrl_snake}/ros2_qt_gui.py"
    if install_marker not in content:
        install_block = f"""

install(PROGRAMS
  ${{PROJECT_NAME}}/{ctrl_snake}/ros2_qt_gui.py
  DESTINATION lib/${{PROJECT_NAME}}
  RENAME {controller_class}QT
)
"""
        content = content + install_block

    write_file(cmake_path, content)


def update_plugin_xml(controller_class: str) -> None:
    xml_path = PACKAGE_PATH / f"{PACKAGE_DIR}_plugin.xml"
    content = read_file(xml_path)

    class_entry = (
        f"  <class name=\"{PACKAGE_DIR}/{controller_class}\"\n"
        f"         type=\"{PACKAGE_DIR}::{controller_class}\" base_class_type=\"controller_interface::ControllerInterface\">\n"
        f"    <description>Auto-generated controller</description>\n"
        f"  </class>\n"
    )

    if class_entry in content:
        return

    if "</library>" not in content:
        print(f"Error: Could not find </library> in {xml_path}")
        sys.exit(1)

    content = content.replace("</library>", class_entry + "</library>")
    write_file(xml_path, content)


def append_to_controllers_yaml(ctrl_name_snake: str, controller_class: str) -> None:
    yaml_path = PACKAGE_PATH / "config" / "controllers.yaml"

    manager_block = f"""
/**:
  controller_manager:
    ros__parameters:
      {ctrl_name_snake}:
        type: {PACKAGE_DIR}/{controller_class}
"""
    params_block = f"""
/**:
  {ctrl_name_snake}:
    ros__parameters:
      arm_id: "fr3"
"""
    
    append_file(yaml_path, manager_block)
    append_file(yaml_path, params_block)


def _prepare_template(raw: str) -> str:
    s = raw
    # escape all remaining braces for .format
    s = s.replace("{", "{{").replace("}", "}}")
    s = s.replace("___PKG___", "{0}")
    s = s.replace("___CLASS___", "{1}")
    s = s.replace("___SNAKE___", "{2}")

    return s

def generate_qt_gui_file(ctrl_snake: str) -> None:

    gui_dir = PACKAGE_PATH / PACKAGE_DIR / ctrl_snake
    gui_path = gui_dir / "ros2_qt_gui.py"

    gui_tpl_raw = """#!/usr/bin/env python3
import sys
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QGridLayout,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
)
from PyQt5.QtCore import QTimer

CONTROL_MODE_TOPIC = "/fr3_1/___CTRL_SNAKE___/control_mode"
JOINT_STATES_TOPIC = "/fr3_1/joint_states"
EE_POSE_TOPIC = "/fr3_1/franka_robot_state_broadcaster/current_pose"


class Fr3QtRosNode(Node):
    def __init__(self) -> None:
        super().__init__("fr3_qt_gui___CTRL_SNAKE___")

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publisher: control mode
        self.control_mode_pub = self.create_publisher(
            Int32,
            CONTROL_MODE_TOPIC,
            10,
        )

        # Subscriber: joint states
        self.joint_states_sub = self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self.joint_states_callback,
            sensor_qos,
        )

        # Subscriber: EE pose
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            EE_POSE_TOPIC,
            self.ee_pose_callback,
            sensor_qos,
        )

        self.latest_joint_data: Dict[str, Tuple[float, float, float]] = {}
        self.latest_ee_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.latest_ee_orientation: Tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        )

    def joint_states_callback(self, msg: JointState) -> None:
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        target_joint_names = ["fr3_joint{}".format(i) for i in range(1, 8)]

        for jname in target_joint_names:
            if jname in name_to_index:
                idx = name_to_index[jname]
                pos = msg.position[idx] if len(msg.position) > idx else 0.0
                vel = msg.velocity[idx] if len(msg.velocity) > idx else 0.0
                eff = msg.effort[idx] if len(msg.effort) > idx else 0.0
                self.latest_joint_data[jname] = (pos, vel, eff)

    def ee_pose_callback(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.latest_ee_position = (p.x, p.y, p.z)

        q = msg.pose.orientation
        self.latest_ee_orientation = (q.x, q.y, q.z, q.w)

    def publish_control_mode(self, value: int) -> None:
        msg = Int32()
        msg.data = value
        self.control_mode_pub.publish(msg)
        self.get_logger().info("Published control_mode: {}".format(value))


class Fr3QtGui(QWidget):
    def __init__(self, ros_node: Fr3QtRosNode) -> None:
        super().__init__()
        self.ros_node = ros_node

        self.joint_labels: Dict[str, Dict[str, QLabel]] = {}
        self.ee_pos_labels: Dict[str, QLabel] = {}
        self.ee_ori_labels: Dict[str, QLabel] = {}

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_from_ros)
        self.timer.start(100)

    def init_ui(self) -> None:
        self.setWindowTitle("FR3 Control & Monitor GUI (___CTRL_SNAKE___)")

        # --- Control mode group ---
        control_group = QGroupBox("Control Mode Sender")
        control_layout = QHBoxLayout()

        control_label = QLabel("Control mode (Int32):")
        self.control_mode_input = QLineEdit()
        send_button = QPushButton("Send")
        self.control_mode_status_label = QLabel("")

        send_button.clicked.connect(self.on_send_control_mode_clicked)

        control_layout.addWidget(control_label)
        control_layout.addWidget(self.control_mode_input)
        control_layout.addWidget(send_button)
        control_layout.addWidget(self.control_mode_status_label)

        control_group.setLayout(control_layout)

        # --- Joint states group ---
        joint_group = QGroupBox("FR3 Joint States (fr3_joint1..7)")
        joint_layout = QGridLayout()

        joint_layout.addWidget(QLabel("Joint"), 0, 0)
        joint_layout.addWidget(QLabel("Position"), 0, 1)
        joint_layout.addWidget(QLabel("Velocity"), 0, 2)
        joint_layout.addWidget(QLabel("Effort"), 0, 3)

        row = 1
        for joint_idx in range(1, 8):
            joint_name = "fr3_joint{}".format(joint_idx)

            name_label = QLabel(joint_name)
            pos_label = QLabel("0.0")
            vel_label = QLabel("0.0")
            eff_label = QLabel("0.0")

            self.joint_labels[joint_name] = {
                "pos": pos_label,
                "vel": vel_label,
                "eff": eff_label,
            }

            joint_layout.addWidget(name_label, row, 0)
            joint_layout.addWidget(pos_label, row, 1)
            joint_layout.addWidget(vel_label, row, 2)
            joint_layout.addWidget(eff_label, row, 3)
            row += 1

        joint_group.setLayout(joint_layout)

        # --- EE pose group ---
        ee_group = QGroupBox("End-Effector Pose")
        ee_layout = QGridLayout()

        ee_layout.addWidget(QLabel("Component"), 0, 0)
        ee_layout.addWidget(QLabel("Value"), 0, 1)

        pos_components = ["x", "y", "z"]
        row = 1
        for comp in pos_components:
            label_name = QLabel("Position {}:".format(comp))
            value_label = QLabel("0.0")
            self.ee_pos_labels[comp] = value_label
            ee_layout.addWidget(label_name, row, 0)
            ee_layout.addWidget(value_label, row, 1)
            row += 1

        spacer = QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)
        ee_layout.addItem(spacer, row, 0)
        row += 1

        ori_components = ["x", "y", "z", "w"]
        for comp in ori_components:
            label_name = QLabel("Orientation {}:".format(comp))
            value_label = QLabel("0.0")
            self.ee_ori_labels[comp] = value_label
            ee_layout.addWidget(label_name, row, 0)
            ee_layout.addWidget(value_label, row, 1)
            row += 1

        ee_group.setLayout(ee_layout)

        # --- Main layout ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(control_group)
        main_layout.addWidget(joint_group)
        main_layout.addWidget(ee_group)

        self.setLayout(main_layout)
        self.resize(800, 400)

    def on_send_control_mode_clicked(self) -> None:
        text = self.control_mode_input.text().strip()
        if text == "":
            self.control_mode_status_label.setText("Ignored: empty input")
            return

        try:
            value = int(text)
        except ValueError:
            # int가 아닌 이상한 값이면 무시
            self.control_mode_status_label.setText("Ignored: not an integer")
            return

        self.ros_node.publish_control_mode(value)
        self.control_mode_status_label.setText("Sent: {}".format(value))

    def update_from_ros(self) -> None:
        # Joint states
        for joint_name, labels in self.joint_labels.items():
            if joint_name in self.ros_node.latest_joint_data:
                pos, vel, eff = self.ros_node.latest_joint_data[joint_name]
                labels["pos"].setText("{:.4f}".format(pos))
                labels["vel"].setText("{:.4f}".format(vel))
                labels["eff"].setText("{:.4f}".format(eff))

        # EE position
        px, py, pz = self.ros_node.latest_ee_position
        self.ee_pos_labels["x"].setText("{:.4f}".format(px))
        self.ee_pos_labels["y"].setText("{:.4f}".format(py))
        self.ee_pos_labels["z"].setText("{:.4f}".format(pz))

        # EE orientation (quaternion)
        ox, oy, oz, ow = self.ros_node.latest_ee_orientation
        self.ee_ori_labels["x"].setText("{:.4f}".format(ox))
        self.ee_ori_labels["y"].setText("{:.4f}".format(oy))
        self.ee_ori_labels["z"].setText("{:.4f}".format(oz))
        self.ee_ori_labels["w"].setText("{:.4f}".format(ow))


def main() -> None:
    rclpy.init(args=sys.argv)
    ros_node = Fr3QtRosNode()
    app = QApplication(sys.argv)

    gui = Fr3QtGui(ros_node)
    gui.show()

    ros_timer = QTimer()
    ros_timer.timeout.connect(lambda: rclpy.spin_once(ros_node, timeout_sec=0.01))
    ros_timer.start(10)

    exit_code = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
"""

    gui_content = gui_tpl_raw.replace("___CTRL_SNAKE___", ctrl_snake)
    write_file(gui_path, gui_content)


def generate_from_templates(controller_class: str) -> None:
    snake = to_snake(controller_class)
    new_h = PACKAGE_PATH / "include" / PACKAGE_DIR / f"{snake}.h"
    new_cpp = PACKAGE_PATH / "src" / f"{snake}.cpp"

    header_tpl_raw = """#pragma once

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
    inline constexpr const char* RESET = "\033[0m";
    inline constexpr const char* BLUE  = "\033[34m"; // Info
    inline constexpr const char* YELLOW= "\033[33m"; // Warn
    inline constexpr const char* RED   = "\033[31m"; // Error
}

#define LOGI(node, fmt, ...) RCLCPP_INFO((node)->get_logger(),  (std::string(ConsoleColor::BLUE)   + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)
#define LOGW(node, fmt, ...) RCLCPP_WARN((node)->get_logger(),  (std::string(ConsoleColor::YELLOW) + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)
#define LOGE(node, fmt, ...) RCLCPP_ERROR((node)->get_logger(), (std::string(ConsoleColor::RED)    + fmt + ConsoleColor::RESET).c_str(), ##__VA_ARGS__)

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using namespace Eigen;

namespace ___PKG___ 
{
class ___CLASS___ : public controller_interface::ControllerInterface 
{
    public:
        ~___CLASS___() override;
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
                joint_ctrl_home_1,
                joint_ctrl_home_2,
                hw8_1,
                hw8_2,
                hw8_3,
                hw8_4,
                hw8_5,
                hw8_6,
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
            void HW8_2(const Vector7d& q_target, const Vector7d& qdot_target, const double duration);
            void HW8_3(const Vector7d& q_target, const Vector7d& qdot_target, const double duration);
            void HW8_4(const Matrix4d x_target, const double duration);
            void HW8_5(const Matrix4d x_target, const double duration);
            void HW8_6(const Matrix4d x_target, const double duration);
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
}  // namespace ___PKG___

"""

    source_tpl_raw = """#include "___PKG___/___SNAKE___.h"
#include <___PKG___/robot_utils.hpp>

namespace ___PKG___ 
{
    // ========================================================================
    // ============================ Core Functions ============================
    // ========================================================================
    ___CLASS___::~___CLASS___()
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

    CallbackReturn ___CLASS___::on_init() 
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
                "___SNAKE___/control_mode",
                rclcpp::QoS(10),
                std::bind(&___CLASS___::keyMapping, this, std::placeholders::_1)
            );

            if (!compute_thread_.joinable())
            {
                std::lock_guard<std::mutex> lock(compute_cv_mutex_);
                stop_compute_thread_ = false;
                compute_requested_ = false;
                compute_completed_ = false;
                compute_thread_ = std::thread(&___CLASS___::computeWorkerLoop, this);
            }

        } 
        catch (const std::exception& e) 
        {
            LOGE(get_node(), "Exception during initialization: %s", e.what());
            return CallbackReturn::ERROR;
        }
        return CallbackReturn::SUCCESS;
    }

    controller_interface::InterfaceConfiguration ___CLASS___::command_interface_configuration() const 
    {
        controller_interface::InterfaceConfiguration command_interface_config;
        command_interface_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
        for (int i = 1; i <= num_joints; ++i) 
        {
            command_interface_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
        }
        return command_interface_config;
    }

    controller_interface::InterfaceConfiguration ___CLASS___::state_interface_configuration() const 
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

    CallbackReturn ___CLASS___::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) 
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

    CallbackReturn ___CLASS___::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) 
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

    controller_interface::CallbackReturn ___CLASS___::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) 
    {
        franka_robot_model_->release_interfaces();
        return CallbackReturn::SUCCESS;
    }

    controller_interface::return_type ___CLASS___::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period)
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

    void ___CLASS___::keyMapping(const std_msgs::msg::Int32& msg)
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

    void ___CLASS___::compute()
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
    void ___CLASS___::moveJointPositionTorque(const Vector7d& target_position, double duration)
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
    void ___CLASS___::HW8_1()
    {
        tau_desired_ = c_;
    }

    void ___CLASS___::HW8_2(const Vector7d& q_target, const Vector7d& qdot_target, const double duration)
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

    void ___CLASS___::HW8_3(const Vector7d& q_target, const Vector7d& qdot_target, const double duration)
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
    
    void ___CLASS___::HW8_4(const Matrix4d x_target, const double duration)
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

    void ___CLASS___::HW8_5(const Matrix4d x_target, const double duration)
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

    void ___CLASS___::HW8_6(const Matrix4d x_target, const double duration)
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

    void ___CLASS___::setMode(const CTRL_MODE& control_mode)
    {
        std::scoped_lock<std::mutex, std::mutex> lk(robot_data_mutex_, calculation_mutex_);

        control_mode_ = control_mode;
        is_mode_changed_ = true;

        LOGI(get_node(), "Mode changed: %d", static_cast<int>(control_mode_));
    }

    void ___CLASS___::printState()
    {
        static int DBG_CNT = 0;
        if (DBG_CNT++ > 1 / (dt_* 50.))
        {
            DBG_CNT = 0;
            // TODO 6: Extend or modify this for debugging your controller
            std::cout << "\\n\\n------------------------------------------------------------------" << std::endl;
            std::cout << "time     : " << std::fixed << std::setprecision(3) << play_time_ << std::endl;
            std::cout << "q now    :\\t";
            std::cout << std::fixed << std::setprecision(3) << q_.transpose() << std::endl;
            std::cout << "x        :\\n";
            std::cout << std::fixed << std::setprecision(3) << x_ << std::endl;
            std::cout << "x dot    :\\t";
            std::cout << std::fixed << std::setprecision(3) << xdot_.transpose() << std::endl;
            std::cout << "J        :\\n";
            std::cout << std::fixed << std::setprecision(3) << J_ << std::endl;
        }
    }

    Matrix4d ___CLASS___::getLinkPose(const VectorXd& q, const std::string& link_name)
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

    MatrixXd ___CLASS___::getLinkJac(const VectorXd& q, const std::string& link_name)
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

    Matrix4d ___CLASS___::getEEPose(const VectorXd& q)
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

    MatrixXd ___CLASS___::getEEJac(const VectorXd& q)
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

    MatrixXd ___CLASS___::getMassMatrix(const VectorXd& q)
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

    // VectorXd ___CLASS___::getGravityVector(const VectorXd& q)
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

    void ___CLASS___::updateJointStates() 
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

    void ___CLASS___::updateRobotData()
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

    void ___CLASS___::computeWorkerLoop()
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
}  // namespace ___PKG___
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(___PKG___::___CLASS___,
                       controller_interface::ControllerInterface)

        """
    header_tpl = _prepare_template(header_tpl_raw)
    source_tpl = _prepare_template(source_tpl_raw)

    h_content = header_tpl.format(PACKAGE_DIR, controller_class, snake)
    cpp_content = source_tpl.format(PACKAGE_DIR, controller_class, snake)

    write_file(new_h, h_content)
    write_file(new_cpp, cpp_content)

    # update_cmakelists(f"src/{snake}.cpp")
    update_cmakelists(f"src/{snake}.cpp", snake, controller_class)
    update_plugin_xml(controller_class)
    generate_qt_gui_file(snake)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller_name", type=str, 
                                        required=True, 
                                        help="Name of Controller")

    args = parser.parse_args()

    controller_class = args.controller_name
    if not re.match(r"^[A-Z][A-Za-z0-9_]*$", controller_class):
        print("Error: controller_name should be a valid C++ class-like identifier starting with uppercase.")
        sys.exit(1)

    generate_from_templates(controller_class)
    append_to_controllers_yaml(to_snake(controller_class), controller_class)

    print(f"Generated controller: {controller_class}")

if __name__ == "__main__":
    main()

