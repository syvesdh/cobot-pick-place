#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <thread>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/aruco.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"

using namespace std::chrono_literals;

class DetectArucoNode : public rclcpp::Node {
  private:
    cv::Mat image_;
    std::mutex image_mutex_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSubscription_;
    bool is_running_;

    float marker_size_;
    std::string calib_file_;

    void get_new_image_callback(sensor_msgs::msg::Image::SharedPtr msg);
    void process_image();
    void getEulerAngles(const cv::Mat &rvec, double &roll, double &pitch, double &yaw);
  
  public:
    static int encoding_to_mat_type(const std::string & encoding);
    DetectArucoNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~DetectArucoNode();
};

int DetectArucoNode::encoding_to_mat_type(const std::string & encoding){
  if (encoding == "mono8") return CV_8UC1;
  else if (encoding == "bgr8") return CV_8UC3;
  else if (encoding == "mono16") return CV_16SC1;
  else if (encoding == "rgba8") return CV_8UC4;
  else if (encoding == "bgra8") return CV_8UC4;
  else if (encoding == "32FC1") return CV_32FC1;
  else if (encoding == "rgb8") return CV_8UC3;
  else if (encoding == "8UC3") return CV_8UC3;
  else {
    std::cout << "the unknown image type is " << encoding << std::endl;
    throw std::runtime_error("Unsupported encoding type");
  }
}

void DetectArucoNode::get_new_image_callback(sensor_msgs::msg::Image::SharedPtr msg){
  try{
    cv::Mat frame(msg->height, msg->width, DetectArucoNode::encoding_to_mat_type(msg->encoding),
      const_cast<unsigned char *>(msg->data.data()), msg->step);
    if (msg->encoding == "rgb8") {
      cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
    }
    
    std::lock_guard<std::mutex> lock(image_mutex_);
    frame.copyTo(this->image_);
  }
  catch(std::runtime_error &exception){
    std::cout << "Exception in image callback: " << exception.what() << std::endl;
  }
}

void DetectArucoNode::getEulerAngles(const cv::Mat &rvec, double &roll, double &pitch, double &yaw) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    
    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0));
    bool singular = sy < 1e-6;

    if (!singular) {
        roll = atan2(R.at<double>(2,1), R.at<double>(2,2));
        pitch = atan2(-R.at<double>(2,0), sy);
        yaw = atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        roll = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        pitch = atan2(-R.at<double>(2,0), sy);
        yaw = 0;
    }
    
    roll = roll * 180.0 / M_PI;
    pitch = pitch * 180.0 / M_PI;
    yaw = yaw * 180.0 / M_PI;
}

void DetectArucoNode::process_image(){
  std::cout << "=========================================================\n";
  std::cout << "ArUco Pose Estimation started (Marker size: " << marker_size_ << "m).\n";
  std::cout << "Looking for markers with ID 0 and 1...\n";
  std::cout << "Press 'q' to quit.\n";
  std::cout << "=========================================================\n";

  cv::FileStorage fs(calib_file_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
      std::cerr << "Error: Calibration file '" << calib_file_ << "' not found." << std::endl;
      std::cerr << "Please run calibrate_camera_tmros2 first to generate this." << std::endl;
      is_running_ = false;
      rclcpp::shutdown();
      return;
  }
  
  cv::Mat mtx, dist;
  fs["mtx"] >> mtx;
  fs["dist"] >> dist;
  fs.release();
  
  std::cout << "Loaded camera matrix and distortion coefficients from " << calib_file_ << std::endl;

  cv::Ptr<cv::aruco::Dictionary> dictionary;
  cv::Ptr<cv::aruco::DetectorParameters> parameters;
  
  // OpenCV < 4.7 Aruco usage (ROS 2 Humble standard)
  dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  parameters = cv::aruco::DetectorParameters::create();

  float m_half = marker_size_ / 2.0;
  std::vector<cv::Point3f> obj_points = {
      cv::Point3f(-m_half, -m_half, 0),
      cv::Point3f(m_half, -m_half, 0),
      cv::Point3f(m_half, m_half, 0),
      cv::Point3f(-m_half, m_half, 0)
  };

  while(is_running_ && rclcpp::ok()){
    cv::Mat frame;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      if (!image_.empty()) {
        image_.copyTo(frame);
      }
    }

    if (!frame.empty()) {
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners, rejected;
      
      cv::aruco::detectMarkers(gray, dictionary, corners, ids, parameters);

      cv::Mat display_frame = frame.clone();

      if (!ids.empty()) {
          cv::aruco::drawDetectedMarkers(display_frame, corners, ids);

          for (size_t i = 0; i < ids.size(); i++) {
              if (ids[i] == 0 || ids[i] == 1) {
                  cv::Mat rvec, tvec;
                  bool success = cv::solvePnP(obj_points, corners[i], mtx, dist, rvec, tvec);
                  
                  if (success) {
                      cv::drawFrameAxes(display_frame, mtx, dist, rvec, tvec, marker_size_ * 0.5);

                      double tx = tvec.at<double>(0);
                      double ty = tvec.at<double>(1);
                      double tz = tvec.at<double>(2);

                      double roll, pitch, yaw;
                      getEulerAngles(rvec, roll, pitch, yaw);

                      char info_text_pos[100];
                      sprintf(info_text_pos, "ID %d XYZ: %.2f, %.2f, %.2f", ids[i], tx, ty, tz);
                      
                      char info_text_rot[100];
                      sprintf(info_text_rot, "RPY: %.1f, %.1f, %.1f", roll, pitch, yaw);

                      int text_x = (int)corners[i][0].x;
                      int text_y = (int)corners[i][0].y - 20;
                      text_y = std::max(text_y, 40);

                      cv::putText(display_frame, info_text_pos, cv::Point(text_x, text_y), 
                                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                      cv::putText(display_frame, info_text_rot, cv::Point(text_x, text_y + 20), 
                                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
                      
                      std::cout << "Marker " << ids[i] << " -> XYZ: [" << tx << ", " << ty << ", " << tz 
                                << "], RPY: [" << roll << ", " << pitch << ", " << yaw << "]" << std::endl;
                  }
              }
          }
      }

      cv::imshow("ArUco 6D Pose Estimation", display_frame);
    }

    int key = cv::waitKey(30) & 0xFF;
    if (key == 'q') {
      is_running_ = false;
      rclcpp::shutdown();
      break;
    }
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
}

DetectArucoNode::DetectArucoNode(const rclcpp::NodeOptions & options) 
  : Node("detect_aruco_pose_tmros2", options), is_running_(true) {

  this->declare_parameter<double>("marker_size", 0.1);
  this->declare_parameter<std::string>("calib", "camera_calibration.yaml");

  marker_size_ = this->get_parameter("marker_size").as_double();
  calib_file_ = this->get_parameter("calib").as_string();

  imageSubscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    "techman_image", 10, std::bind(&DetectArucoNode::get_new_image_callback, this, std::placeholders::_1));
  
  std::thread(&DetectArucoNode::process_image, this).detach();
}

DetectArucoNode::~DetectArucoNode() {
  is_running_ = false;
}

int main(int argc, char *argv[]){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DetectArucoNode>();
  rclcpp::spin(node);
  std::cout << "end spin" << std::endl;
  
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  return 0;
}
