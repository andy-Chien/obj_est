#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

#include <iostream>

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointTRGB;

std::string open_path_image = "../test_pcl_ws/src/pcl_test/data/save_img/test_img.jpg";   // [1] img_file
std::string save_path_rgb = "../test_pcl_ws/src/pcl_test/data/save_img/rgb_";             // [2] img_rs_rgb; [6] all_rs
std::string save_path_depth = "../test_pcl_ws/src/pcl_test/data/save_img/depth_";         // [3] img_rs_depth; [6] all_rs
std::string open_path_cloud = "../test_pcl_ws/src/pcl_test/data/save_cloud/test_cloud.pcd";// [4] cloud_file
std::string save_path_cloud = "../test_pcl_ws/src/pcl_test/data/save_cloud/cloud_";       // [5] cloud_rs; [6] all_rs

int cnt_rgb = 0;
int cnt_depth = 0;
int cnt_cloud = 0;

void image_cb_rgb(const sensor_msgs::ImageConstPtr &msg);
void image_cb_depth(const sensor_msgs::ImageConstPtr &msg);
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

ros::Publisher pub;

using namespace std;
pcl::PointCloud<PointTRGB>::Ptr cloud(new pcl::PointCloud<PointTRGB>);      //pcl::PointXYZRGB

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  // // Container for original & filtered data
  // pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2; 
  // pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
  // pcl::PCLPointCloud2 cloud_filtered;

  // // Convert to PCL data type
  // pcl_conversions::toPCL(*cloud_msg, *cloud);

  // // Perform the actual filtering
  // pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  // sor.setInputCloud (cloudPtr);
  // sor.setLeafSize (0.1, 0.1, 0.1);
  // sor.filter (cloud_filtered);

  // // Convert to ROS data type
  // sensor_msgs::PointCloud2 output;
  // pcl_conversions::moveFromPCL(cloud_filtered, output);

  // // Publish the data
  // pub.publish (output);

    // 將點雲格式由sensor_msgs/PointCloud2轉成pcl/PointCloud(PointXYZ, PointXYZRGB)
  pcl::fromROSMsg(*cloud_msg, *cloud);

  // Save PointCloud 
  ostringstream os;
  os << cnt_cloud;
  string file_path_cloud = save_path_cloud + os.str();
  file_path_cloud = file_path_cloud + ".pcd";
  
  pcl::io::savePCDFileASCII<PointTRGB>(file_path_cloud, *cloud);
  cout << "Cloud saved: " << file_path_cloud << "; (width, height) = " << cloud->width << ", " << cloud->height << endl;

  cnt_cloud++;
}


int main (int argc, char** argv)
{
  
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;
  ros::NodeHandle n;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/camera/depth/color/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("/process_cloud", 1);
  
  // Spin
  ros::spin ();
}