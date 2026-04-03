#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include "techman_robot_msgs/srv/techman_robot_command.hpp"
#include "tm_msgs/msg/feedback_state.hpp"

using namespace std::chrono_literals;

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("send_command_client");
  auto client = node->create_client<techman_robot_msgs::srv::TechmanRobotCommand>("tm_send_command");

  while(!client->wait_for_service(1s)){
    if(!rclcpp::ok()){
      RCLCPP_ERROR_STREAM(node->get_logger(), "Client interrupted while waiting for service to appear.");
      return 1;
    }
    RCLCPP_INFO_STREAM(node->get_logger(), "waiting for service...");
  }

  bool is_sct_connected = false;
  auto sub = node->create_subscription<tm_msgs::msg::FeedbackState>(
    "feedback_states", 10,
    [&is_sct_connected](const tm_msgs::msg::FeedbackState::SharedPtr msg) {
      is_sct_connected = msg->is_sct_connected;
    });

  // Since TM drops listen to check camera for 500ms intermittently, we block until it reconnects
  RCLCPP_INFO_STREAM(node->get_logger(), "Waiting for Listen Node to be connected...");
  while(rclcpp::ok() && !is_sct_connected) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(10ms);
  }
  
  if (!rclcpp::ok()) return 0;
  
  RCLCPP_INFO_STREAM(node->get_logger(), "Listen Node is connected! Sending command...");

  auto request = std::make_shared<techman_robot_msgs::srv::TechmanRobotCommand::Request>();
  request->command = "MOVE_JOG";
  request->command_parameter_string = "0,0,90,0,90,0";

  auto res_future = client->async_send_request(request);
  auto status = rclcpp::spin_until_future_complete(node, res_future, 2000ms);
  
  if(status == rclcpp::FutureReturnCode::SUCCESS){
    auto res = res_future.get();
    RCLCPP_INFO_STREAM(node->get_logger(), "is_success: " << (res->is_success ? "true" : "false"));	
  } else if(status == rclcpp::FutureReturnCode::TIMEOUT){
    RCLCPP_ERROR_STREAM(node->get_logger(), "Service call completely timed out after 2000ms (2s)!");
  } else {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Service call failed or interrupted.");
  }

  rclcpp::shutdown();
  return 0;
}