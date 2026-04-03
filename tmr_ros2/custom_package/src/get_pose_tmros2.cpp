#include <memory>
#include <vector>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "tm_msgs/msg/feedback_state.hpp"

using std::placeholders::_1;

class GetPoseNode : public rclcpp::Node
{
  public:
    GetPoseNode()
    : Node("get_pose_tmros2")
    {
      RCLCPP_INFO(this->get_logger(), "Starting TM5-700 6D Pose Subscriber...");
      
      subscription_ = this->create_subscription<tm_msgs::msg::FeedbackState>(
      "feedback_states", 10, std::bind(&GetPoseNode::topic_callback, this, _1));
    }

  private:
    void topic_callback(const tm_msgs::msg::FeedbackState::SharedPtr msg) const
    {
      if(msg->tool0_pose.size() == 6 && msg->tool_pose.size() == 6){
        RCLCPP_INFO_STREAM(this->get_logger(), "\n--- TM5-700 Pose Update ---");
        
        // tool0_pose is the flange (end-effector) pose relative to base
        RCLCPP_INFO_STREAM(this->get_logger(), "End-Effector (Flange) 6D Pose:");
        RCLCPP_INFO_STREAM(this->get_logger(), "  XYZ Base : [" << msg->tool0_pose[0] << ", " << msg->tool0_pose[1] << ", " << msg->tool0_pose[2] << "]");
        RCLCPP_INFO_STREAM(this->get_logger(), "  RPY Base : [" << msg->tool0_pose[3] << ", " << msg->tool0_pose[4] << ", " << msg->tool0_pose[5] << "]");

        // tool_pose is the TCP (Tool Center Point) pose relative to base
        // If the TM camera is configured as the active tool in TMflow, this is exactly the camera's pose!
        RCLCPP_INFO_STREAM(this->get_logger(), "TCP / Camera 6D Pose:");
        RCLCPP_INFO_STREAM(this->get_logger(), "  XYZ Base : [" << msg->tool_pose[0] << ", " << msg->tool_pose[1] << ", " << msg->tool_pose[2] << "]");
        RCLCPP_INFO_STREAM(this->get_logger(), "  RPY Base : [" << msg->tool_pose[3] << ", " << msg->tool_pose[4] << ", " << msg->tool_pose[5] << "]");
      }
    }
    
    rclcpp::Subscription<tm_msgs::msg::FeedbackState>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GetPoseNode>());
  rclcpp::shutdown();
  return 0;
}
