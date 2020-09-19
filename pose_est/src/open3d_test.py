#!/usr/bin/env python3
import sys
sys.path.insert(1, "/usr/local/lib/python3.5/dist-packages/cv2")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import cv2, os, sys, getopt
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import time
# import rospy
from os.path import dirname, abspath

Data_Dir = dirname(dirname(abspath(__file__))) + '/PCD_Data'

def setVisual(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()
    # o3d.visualization.draw_geometries([pcd],
    #                               front=[0.0, 0.0, 1.0],
    #                               lookat=[0.0, 0.0, 1.0],
    #                               up=[-0.0694, -0.9768, 1])

def setVisual_2(pcd_1, pcd_2):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_1)
    vis.add_geometry(pcd_2)
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_data, target_data, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = source_data
    target = target_data
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result

def DBSCAN_Clustering(pcd, eps=0.02, min_points=10):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        print("cm ", cm)
    
    max_label = labels.max()
    print("point cloud has {} clusters".format(max_label + 1))
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def Plane_Segmentation(pcd, distance_threshold=0.01):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print("Plane equation: {}x + {}y + {}z + {}".format(a, b, c, d))
    inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    return inlier_cloud, outlier_cloud

#=================methods combined=================
def Global_Registration(source_pcd, target_pcd, voxel_size=10):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_pcd, target_pcd, voxel_size)
    result_global_registration = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
    return result_global_registration

def setTransform(source_pcd, target_pcd, transformation, paintUniformColor = False):
    source_temp = copy.deepcopy(source_pcd)
    target_temp = copy.deepcopy(target_pcd)
    source_temp.transform(transformation)
    if paintUniformColor:
        source_temp.paint_uniform_color([1, 0.706, 0])
        # target_temp.paint_uniform_color([0, 0.651, 0.929])

    final_pcd = source_temp + target_temp

    return final_pcd, source_temp, target_temp

def SegNClustering(source_pcd, distance_threshold=10, Clustering_Group="out", eps=0.02, min_points=10):
    inliner, outliner = Plane_Segmentation(source_pcd, distance_threshold=10)

    if Clustering_Group == "in":
        source_pcd_clustering = DBSCAN_Clustering(inliner, eps=eps, min_points=min_points)
    else:
        source_pcd_clustering = DBSCAN_Clustering(outliner, eps=eps, min_points=min_points)

    return source_pcd_clustering
#=================methods combined=================/

#=================I/O Data=================
source_data = Data_Dir + "/realsense_data/test_1.ply"
target_data = Data_Dir + "/realsense_data/test_2.ply"
target_data_2 = Data_Dir + "/realsense_data/test_3.ply"
metal_bin = Data_Dir + "/cad_data/solomon_part_8000_dn.pcd"

# read pcd or ply data
source_pcd = o3d.io.read_point_cloud(source_data)
target_pcd = o3d.io.read_point_cloud(target_data)
target_pcd_2 = o3d.io.read_point_cloud(target_data_2)
metal_pcd  = o3d.io.read_point_cloud(metal_bin)

# help(target_pcd)

#rescale for the same size 
source_pcd.scale(1000, center = source_pcd.get_center())
target_pcd.scale(1000, center = target_pcd.get_center())
target_pcd_2.scale(1000, center = target_pcd_2.get_center())
# metal_pcd.scale(1000, center = metal_pcd.get_center())
#=================I/O Data=================/

#=================Debug=================
# test switch True False
Test_Clustering = False
Test_GlobalRegistration =True
#=================Debug=================/

# setVisual(metal_pcd)

if __name__ == '__main__':

    if Test_Clustering:
        #down sampling
        downpcd = source_pcd.voxel_down_sample(voxel_size=0.002)

        #Vertex normal estimation
        # downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        reg_result = SegNClustering(source_pcd, distance_threshold=10, eps=20, min_points=10)

        setVisual(reg_result)

    if Test_GlobalRegistration:
        #==============Stage 1================
        voxel_size = 10

        result_global_registration = Global_Registration(source_pcd, target_pcd, voxel_size = voxel_size)
                                                    
        print("result_global_registration ", result_global_registration)
        print("2 ground transformation ", result_global_registration.transformation)

        Stage_1_result_pcd, _, _ = setTransform(source_pcd, target_pcd, result_global_registration.transformation)

        # setVisual(Stage_1_result_pcd)
        #==============Stage 1================/
        
        Stage_1_SegNClustering_result_pcd = SegNClustering(Stage_1_result_pcd, distance_threshold=10, eps=2, min_points=10)
        
        setVisual(Stage_1_SegNClustering_result_pcd)

        #==============Stage 2================
        result_global_registration_2 = Global_Registration(metal_pcd, Stage_1_SegNClustering_result_pcd, voxel_size = voxel_size)

        print(result_global_registration_2)
        print("metal to 2 ground transformation ", result_global_registration_2.transformation)

        Stage_2_result_pcd, _, _ = setTransform(metal_pcd, Stage_1_SegNClustering_result_pcd, result_global_registration_2.transformation, paintUniformColor = True)

        setVisual(Stage_2_result_pcd + Stage_1_result_pcd)
        #==============Stage 2================/
