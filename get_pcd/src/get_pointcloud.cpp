#include <ros/ros.h>
#include <ros/package.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include "pcl_ros/transforms.h"

#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <yaml-cpp/yaml.h>
#include "get_pcd/save_pcd.h"
#include "realsense2_camera/Extrinsics.h"

#define PCDSONCE 5 // PCDSONCE pcd per service call
using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointTRGB;

std::string save_path_cloud = ros::package::getPath("get_pcd") + "/data/save_cloud/";       // [5] cloud_rs; [6] all_rs
std::string file_name = "";
bool save_mix = false;

YAML::Node doc;
boost::thread* save_file_thread;
boost::thread* pcd_sum_thread;
boost::mutex mutex;
string *file_path;
Eigen::MatrixXf curr_trans = Eigen::MatrixXf::Identity(4,4);
Eigen::MatrixXf d2c_trans = Eigen::MatrixXf::Identity(4,4);

pcl::PointCloud<PointTRGB>::Ptr cloud_mix(new pcl::PointCloud<PointTRGB>);
pcl::PointCloud<PointTRGB>::Ptr cloud(new pcl::PointCloud<PointTRGB>);

int cnt = 0;
int mix_cnt = 0;
void set_d2color_trans(const realsense2_camera::Extrinsics &trans)
{
  for(int i=0; i<3; i++)
  {
    for(int j=0; j<3; j++)
    {
      d2c_trans(i, j) = trans.rotation[i*3+j];
    }
    d2c_trans(i, 3) = trans.translation[i];
  }
  cout << d2c_trans << endl;
}

void save_pcd(pcl::PointCloud<PointTRGB>::Ptr pcd, string* path)
{
  pcl::io::savePCDFileASCII<PointTRGB>(*path, *pcd);
  cout << "Cloud saved: " << *path << "; (width, height) = " << pcd->width << ", " << pcd->height << endl;
  return;
}

void add_two_pcd(pcl::PointCloud<PointTRGB>::Ptr a, pcl::PointCloud<PointTRGB>::Ptr b)
{
  *a = *a + *b;
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  pcl::fromROSMsg(*cloud_msg, *cloud);
  return;
}

bool get_pcd_callback (get_pcd::save_pcd::Request &req, get_pcd::save_pcd::Response &res)
{
  file_name = req.name;
  save_mix = req.save_mix;
  
  if(!req.curr_trans.empty())
  {
    for(int i=0; i<3; i++)
    {
      for(int j=0; j<3; j++)
      {
        curr_trans(i, j) = req.curr_trans[i*3+j];
      }
      curr_trans(i, 3) = req.curr_trans[i];
    }
  }
  pcl::PointCloud<PointTRGB>::Ptr cloud_base(new pcl::PointCloud<PointTRGB>);
  pcl::PointCloud<PointTRGB>::Ptr cloud_filtered(new pcl::PointCloud<PointTRGB>);
  pcl::VoxelGrid<PointTRGB> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.0015f, 0.0015f, 0.0015f);
  sor.filter (*cloud_filtered);
  Eigen::Matrix4f matrix = curr_trans * d2c_trans;
  pcl::transformPointCloud(*cloud_filtered, *cloud_base, matrix);

  if(cloud_mix->empty())
  {
    *cloud_mix =  *cloud_base;
  }
  else
  {
    // pcd_sum_thread = new boost::thread(boost::bind(&add_two_pcd, cloud_mix, cloud_base));
    add_two_pcd(cloud_mix, cloud_base);
    // *cloud_mix =  *cloud_mix + *cloud_base;
  }
  file_path = new string(save_path_cloud + file_name + std::to_string(cnt) + ".pcd");
  // ========go thread========
  // save_file_thread = new boost::thread(boost::bind(&save_pcd, cloud_base, file_path));
  // =======no thread=======
  pcl::io::savePCDFileASCII<PointTRGB>(*file_path, *cloud_base);
  cout << "Cloud saved: " << *file_path << "; (width, height) = " << cloud_base->width << ", " << cloud_base->height << endl;
  // pcd_sum_thread->join();
  // save_file_thread->join();
  if(save_mix)
  {
    pcl::PointCloud<PointTRGB>::Ptr mix_save(new pcl::PointCloud<PointTRGB>);
    sor.setInputCloud (cloud_mix);
    sor.setLeafSize (0.0015f, 0.0015f, 0.0015f);
    sor.filter (*mix_save);
    string file_path_mix = string(save_path_cloud + "cloud_mix_" + std::to_string(mix_cnt) + ".pcd");
    pcl::io::savePCDFileASCII<PointTRGB>(file_path_mix, *mix_save);
    cout << "Cloud saved: " << file_path_mix << "; (width, height) = " << cloud_mix->width << ", " << cloud_mix->height << endl;
    mix_cnt++;
    save_mix = false;
  }
  cnt++;
  res.done = true;
  return true;
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle n;
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber d_to_color_sub = n.subscribe("/camera/extrinsics/depth_to_color", 1, &set_d2color_trans);
  ros::Subscriber point_sub = n.subscribe("/camera/depth_registered/points", 1, &cloud_cb);
  ros::ServiceServer get_pc_service = n.advertiseService("/get_pcd", &get_pcd_callback);
  ros::spin ();
}
