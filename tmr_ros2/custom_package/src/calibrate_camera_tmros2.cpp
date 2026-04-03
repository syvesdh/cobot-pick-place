#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <thread>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"

class CalibrateCameraNode : public rclcpp::Node {
  private:
    cv::Mat image_;
    std::mutex image_mutex_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSubscription_;
    bool is_running_;

    // Calibration variables
    cv::Size board_size_ = cv::Size(9, 6);
    float square_size_;
    std::string output_file_;
    
    std::vector<cv::Point3f> objp_;
    std::vector<std::vector<cv::Point3f>> objpoints_;
    std::vector<std::vector<cv::Point2f>> imgpoints_;
    
    int frames_captured_ = 0;
    std::chrono::time_point<std::chrono::system_clock> last_capture_time_;
    const double capture_interval_ = 1.0; // seconds

    void get_new_image_callback(sensor_msgs::msg::Image::SharedPtr msg);
    void process_image();
  
  public:
    static int encoding_to_mat_type(const std::string & encoding);
    CalibrateCameraNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~CalibrateCameraNode();
};

int CalibrateCameraNode::encoding_to_mat_type(const std::string & encoding){
  if (encoding == "mono8") {
    return CV_8UC1;
  } else if (encoding == "bgr8") {
    return CV_8UC3;
  } else if (encoding == "mono16") {
    return CV_16SC1;
  } else if (encoding == "rgba8") {
    return CV_8UC4;
  } else if (encoding == "bgra8") {
    return CV_8UC4;
  } else if (encoding == "32FC1") {
    return CV_32FC1;
  } else if (encoding == "rgb8") {
    return CV_8UC3;
  } else if (encoding =="8UC3"){
    return CV_8UC3;
  } else {
    std::cout << "the unknown image type is " << encoding << std::endl;
    throw std::runtime_error("Unsupported encoding type");
  }
}

void CalibrateCameraNode::get_new_image_callback(sensor_msgs::msg::Image::SharedPtr msg){
  try{
    cv::Mat frame(msg->height, msg->width, CalibrateCameraNode::encoding_to_mat_type(msg->encoding),
      const_cast<unsigned char *>(msg->data.data()), msg->step);
    if (msg->encoding == "rgb8") {
      cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
    }
    
    std::lock_guard<std::mutex> lock(image_mutex_);
    frame.copyTo(this->image_);
  }
  catch(std::runtime_error &exception){
    std::cout << "there is a exception " << exception.what() << std::endl;
  }
}

void CalibrateCameraNode::process_image(){
  std::cout << "=========================================================\n";
  std::cout << "Camera calibration started.\n";
  std::cout << "Moving the 9x6 chessboard in front of the camera.\n";
  std::cout << "The script will automatically capture a frame every 1 second\n";
  std::cout << "when it detects all the chessboard corners.\n";
  std::cout << "Press 'c' when you have captured enough frames (e.g., 20+).\n";
  std::cout << "Press 'q' at any time to quit without saving.\n";
  std::cout << "=========================================================\n";

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

      std::vector<cv::Point2f> corners;
      bool ret_corners = cv::findChessboardCorners(gray, board_size_, corners, 
                                                   cv::CALIB_CB_ADAPTIVE_THRESH | 
                                                   cv::CALIB_CB_FAST_CHECK | 
                                                   cv::CALIB_CB_NORMALIZE_IMAGE);

      cv::Mat display_frame = frame.clone();

      if (ret_corners) {
        cv::drawChessboardCorners(display_frame, board_size_, corners, ret_corners);

        auto current_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = current_time - last_capture_time_;

        if (elapsed.count() >= capture_interval_) {
          cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

          objpoints_.push_back(objp_);
          imgpoints_.push_back(corners);

          frames_captured_++;
          last_capture_time_ = current_time;
          std::cout << "Captured frame " << frames_captured_ << ". Change the angle/position of the board." << std::endl;

          cv::bitwise_not(display_frame, display_frame);
        }
      }

      cv::putText(display_frame, "Captured: " + std::to_string(frames_captured_), 
                  cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      cv::putText(display_frame, "Press 'c' to calculate & save", 
                  cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

      cv::imshow("Camera Calibration", display_frame);
    }

    int key = cv::waitKey(30) & 0xFF;
    if (key == 'q') {
      std::cout << "Exiting without calibration." << std::endl;
      is_running_ = false;
      rclcpp::shutdown();
      break;
    } else if (key == 'c') {
      if (frames_captured_ < 5) {
        std::cout << "Warning: Only " << frames_captured_ << " frames captured. Calibration might be inaccurate. Need at least 5-10." << std::endl;
      } else {
        is_running_ = false;
        break;
      }
    }
  }

  if (!is_running_ && frames_captured_ == 0) {
      // Exited early.
  } else if (frames_captured_ > 0) {
    std::cout << "\nCalculating camera calibration... Please wait." << std::endl;
    cv::Mat mtx, dist;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Size img_size;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      if(!image_.empty()){
         img_size = cv::Size(image_.cols, image_.rows);
      } else {
         img_size = cv::Size(640, 480);
      }
    }
    
    double ret = cv::calibrateCamera(objpoints_, imgpoints_, img_size, mtx, dist, rvecs, tvecs);

    if (ret > 0) {
      std::cout << "Calibration successful! RMS re-projection error: " << ret << std::endl;
      std::cout << "\nCamera Matrix:\n" << mtx << std::endl;
      std::cout << "\nDistortion Coefficients:\n" << dist << std::endl;

      cv::FileStorage fs(output_file_, cv::FileStorage::WRITE);
      fs << "mtx" << mtx;
      fs << "dist" << dist;
      fs.release();

      std::cout << "\nSaved calibration parameters to " << output_file_ << std::endl;
    } else {
      std::cout << "Calibration failed." << std::endl;
    }
  } else {
    std::cout << "No valid frames were captured." << std::endl;
  }
  
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
}

CalibrateCameraNode::CalibrateCameraNode(const rclcpp::NodeOptions & options) 
  : Node("calibrate_camera_tmros2", options), is_running_(true) {

  this->declare_parameter<double>("square_size", 0.025);
  this->declare_parameter<std::string>("output", "camera_calibration.yaml");

  square_size_ = this->get_parameter("square_size").as_double();
  output_file_ = this->get_parameter("output").as_string();

  for (int i = 0; i < board_size_.height; i++) {
    for (int j = 0; j < board_size_.width; j++) {
      objp_.push_back(cv::Point3f(j * square_size_, i * square_size_, 0.0f));
    }
  }

  last_capture_time_ = std::chrono::system_clock::now();

  imageSubscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    "techman_image", 10, std::bind(&CalibrateCameraNode::get_new_image_callback, this, std::placeholders::_1));
  
  std::thread(&CalibrateCameraNode::process_image, this).detach();
}

CalibrateCameraNode::~CalibrateCameraNode() {
  is_running_ = false;
}

int main(int argc, char *argv[]){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CalibrateCameraNode>();
  rclcpp::spin(node);
  std::cout << "end spin" << std::endl;
  
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  return 0;
}
