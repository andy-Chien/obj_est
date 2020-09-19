#include <ros/ros.h>
#include <ros/package.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <string>
#include "get_pcd/save_pcd.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <boost/thread.hpp>
#include <yaml-cpp/yaml.h>
#define PCDSONCE 5 // PCDSONCE pcd per service call
using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointTRGB;

std::string save_path_cloud = ros::package::getPath("get_pcd") + "/data/save_cloud/";       // [5] cloud_rs; [6] all_rs
std::string file_name = "";
bool do_save = false;

YAML::Node doc;
boost::thread* save_file_thread;
string *file_path;
Eigen::MatrixXd curr_trans = Eigen::MatrixXd::Identity(4,4);

int cnt = PCDSONCE * 10;

void save_pcd(pcl::PointCloud<PointTRGB>::Ptr pcd, string* path)
{
  pcl::io::savePCDFileASCII<PointTRGB>(*path, *pcd);
  cout << "Cloud saved: " << *path << "; (width, height) = " << pcd->width << ", " << pcd->height << endl;
  return;
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  if(cnt > PCDSONCE * 3)  
    return;
  pcl::PointCloud<PointTRGB>::Ptr cloud(new pcl::PointCloud<PointTRGB>);  
  pcl::fromROSMsg(*cloud_msg, *cloud);
  if(cnt%3 == 0) //save one pcd every three times
  {
    // ========go thread========
    file_path = new string;
    *file_path = save_path_cloud + file_name + std::to_string(cnt) + ".pcd";
    save_file_thread = new boost::thread(boost::bind(&save_pcd, cloud, file_path));
    delete save_file_thread;
    // =======no thread=======
    // string file_path_cloud = save_path_cloud + file_name + std::to_string(cnt) + ".pcd";
    // cnt ++;
    // pcl::io::savePCDFileASCII<PointTRGB>(file_path_cloud, *cloud);
    // cout << "Cloud saved: " << file_path_cloud << "; (width, height) = " << cloud->width << ", " << cloud->height << endl;
  }
  cnt++;
}

bool get_pcd_callback (get_pcd::save_pcd::Request &req, get_pcd::save_pcd::Response &res)
{
  file_name = req.name;
  do_save = true;
  res.done = true;
  cnt = 0;
  return true;
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle n;
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber point_sub = n.subscribe("/camera/depth_registered/points", 1, &cloud_cb);
  ros::ServiceServer service = n.advertiseService("/get_pcd", &get_pcd_callback);
  ros::spin ();
}

// *a =*a +*b;
// Eigen::Matrix4f matrix =Eigen::Matrix4f::Identity();
// pcl::transformPointCloud(*a, *a_trans, matrix);