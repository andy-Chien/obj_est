#!/usr/bin/env python3
import sys
# sys.path.insert(1, "/usr/local/lib/python3.5/dist-packages/cv2")
# sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import cv2, os, sys, getopt
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import time
import rospy
import rospkg
from os.path import dirname, abspath

# Data_Dir = dirname(dirname(abspath(__file__))) + '/PCD_Data'
rospack = rospkg.RosPack()
Data_Dir = rospack.get_path('get_pcd') + '/data/save_cloud'
Config_Dir = rospack.get_path('pose_est') + '/config'
Center = [0.198, 0.282, - 0.24]
Width = [0.36, 0.56, 0.06]

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
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_data, target_data, voxel_size):
    # print(":: Load two point clouds and disturb initial pose.")
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
    distance_threshold = voxel_size * 3
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(True), 6, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(40000000, 50000))
    # result = o3d.registration.registration_ransac_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
    #     o3d.registration.TransformationEstimationPointToPlane(), 6, [
    #         o3d.registration.CorrespondenceCheckerBasedOnDistance(
    #             distance_threshold)
    #     ], o3d.registration.RANSACConvergenceCriteria(8000000, 1000))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 1
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.6f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result

def DBSCAN_Clustering(pcd, eps=0.02, min_points=10):
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        # print("cm ", cm)
    
    max_label = labels.max()
    # print("point cloud has {} clusters".format(max_label + 1))
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def Plane_Segmentation(pcd, distance_threshold=0.01):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    # print("Plane equation: {}x + {}y + {}z + {}".format(a, b, c, d))
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

def execute_refine_Registration(source_pcd, target_pcd, result_ransac, voxel_size=10):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_pcd, target_pcd, voxel_size)
    result_global_registration = refine_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size, result_ransac)
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
    # print('fuckfuck')
    inliner, outliner = Plane_Segmentation(source_pcd, distance_threshold=10)
    # print('fuckfuck')
    print(len(inliner.points))
    print(len(outliner.points))
    if Clustering_Group == "in":
        source_pcd_clustering = DBSCAN_Clustering(inliner, eps=eps, min_points=min_points)
    else:
        source_pcd_clustering = DBSCAN_Clustering(outliner, eps=eps, min_points=min_points)
    # print('fuckfuck')
    return source_pcd_clustering

def crop_point_cloud(pcd):
    pts = np.asarray(pcd.points)
    p_colors = np.asarray(pcd.colors)
    temp = pts[:, 0]
    valid_idx = np.where(abs(temp - Center[0]) < Width[0]/2)
    pts = pts[valid_idx]
    p_colors = p_colors[valid_idx]
    temp = pts[:, 1]
    valid_idx = np.where(abs(temp - Center[1]) < Width[1]/2)
    pts = pts[valid_idx]
    p_colors = p_colors[valid_idx]
    temp = pts[:, 2]
    valid_idx = np.where(abs(temp - Center[2]) < Width[2]/2)
    pcd.points = o3d.utility.Vector3dVector(pts[valid_idx])
    pcd.colors = o3d.utility.Vector3dVector(p_colors[valid_idx])
    return pcd

def delete_point_cloud(s_pcd, t_pcd):
    t_pts = np.asarray(t_pcd.points)
    pts = np.asarray(s_pcd.points)
    p_colors = np.asarray(s_pcd.colors)
    p_normals = np.asarray(s_pcd.normals)
    for t_pt in t_pts:
        # print('t_pt\n',t_pt)
        # print(pts.shape, t_pt.shape, np.repeat(t_pt, len(pts), axis=0).shape)
        temp = pts[:,:] - np.repeat([t_pt], len(pts), axis=0)
        valid_idx = np.where(np.linalg.norm(temp, axis=1) > 0.01)
        # print(len(valid_idx))
        pts = pts[valid_idx]
        p_colors = p_colors[valid_idx]
        p_normals = p_normals[valid_idx]
    s_pcd.points = o3d.utility.Vector3dVector(pts)
    s_pcd.colors = o3d.utility.Vector3dVector(p_colors)
    s_pcd.normals = o3d.utility.Vector3dVector(p_normals)
    return s_pcd

#=================methods combined=================/

#=================I/O Data================

# target_pcd_2 = o3d.io.read_point_cloud(target_data_2).voxel_down_sample(voxel_size=0.002)
# metal_pcd  = o3d.io.read_point_cloud(metal_bin).voxel_down_sample(voxel_size=0.002)

# help(target_pcd)

#rescale for the same size 
# source_pcd.scale(1000, center = source_pcd.get_center())
# target_pcd.scale(1000, center = target_pcd.get_center())
# target_pcd_2.scale(1000, center = target_pcd_2.get_center())
# metal_pcd.scale(1000, center = metal_pcd.get_center())
#=================I/O Data=================/

#=================Debug=================
# test switch True False
Test_Clustering = False
Test_GlobalRegistration = False
Test_Detect = True
#=================Debug=================/

# setVisual(metal_pcd)

if __name__ == '__main__':
    if Test_Clustering:
        # pcd = o3d.io.read_point_cloud(Data_Dir + "/test2/sampl_from_mesh.pcd").voxel_down_sample(voxel_size=0.002)
        # pts = np.asarray(pcd.points)
        # p_colors = np.asarray(pcd.colors)
        # p_normals = np.asarray(pcd.normals)
        # temp = pts[:, 0]
        # # valid_idx = np.where(abs(temp - 0.192) < 0.046/2)
        # valid_idx = np.where(abs(temp - 0.194) < 0.046/2)
        # pts = pts[valid_idx]
        # p_colors = p_colors[valid_idx]
        # p_normals = p_normals[valid_idx]
        # temp = pts[:, 1]
        # # valid_idx = np.where(abs(temp - 0.335) < 0.088/2)
        # valid_idx = np.where(abs(temp - 0.216) < 0.088/2)
        # pts = pts[valid_idx]
        # p_colors = p_colors[valid_idx]
        # p_normals = p_normals[valid_idx]
        # temp = pts[:, 2]
        # # valid_idx = np.where(abs(temp - (-0.246)) < 0.01/2)
        # valid_idx = np.where(abs(temp - (-0.25)) < 0.01/2)
        # pcd.points = o3d.utility.Vector3dVector(pts[valid_idx])
        # pcd.colors = o3d.utility.Vector3dVector(p_colors[valid_idx])
        # pcd.normals = o3d.utility.Vector3dVector(p_normals[valid_idx])
        # o3d.io.write_point_cloud(Data_Dir + "/model/target_from_mesh_2.pcd", pcd)


        pcd = o3d.io.read_point_cloud(Data_Dir + "/model/target_from_mesh_2.pcd")
        transformation = np.asarray([[1.0, 0.0, 0.0, -0.194], 
                                     [0.0, 1.0, 0.0, -0.216],
                                     [0.0, 0.0, 1.0, 0.25], 
                                     [0.0, 0.0, 0.0, 1.0]])
        pcd.transform(transformation)
        o3d.io.write_point_cloud(Data_Dir + "/model/target_from_mesh_2_trans.pcd", pcd)
        pcd = o3d.io.read_point_cloud(Data_Dir + "/model/target_from_mesh_1.pcd")
        transformation = np.asarray([[1.0, 0.0, 0.0, -0.192], 
                                     [0.0, 1.0, 0.0, -0.335],
                                     [0.0, 0.0, 1.0, 0.246], 
                                     [0.0, 0.0, 0.0, 1.0]])
        pcd.transform(transformation)
        o3d.io.write_point_cloud(Data_Dir + "/model/target_from_mesh_1_trans.pcd", pcd)
        
    if Test_Detect:
        target_data = Data_Dir + "/test2/mdfk_mix.pcd"
        source_datas = []
        for i in range(2):
            source_data = Data_Dir + "/model/target_from_mesh_" + str(i+1) + "_trans.pcd"
            source_datas.append(source_data)

        # target_data_2 = Data_Dir + "/blue_light/mdfk2.pcd"
        # metal_bin = Data_Dir + "/part_model/solomon_part_12000.pcd"

        # read pcd or ply data
        target_pcd = o3d.io.read_point_cloud(target_data).voxel_down_sample(voxel_size=0.002)
        source_pcds = []
        for i in range(2):
            source_pcd = o3d.io.read_point_cloud(source_datas[i]).voxel_down_sample(voxel_size=0.002)
            source_pcds.append(source_pcd)
        transed_pcds = []
        result_pcd = None


        pts = np.asarray(target_pcd.points)
        p_colors = np.asarray(target_pcd.colors)
        temp = pts[:, 0]
        valid_idx = np.where(abs(temp - Center[0]) < Width[0]/2-0.01)
        pts = pts[valid_idx]
        p_colors = p_colors[valid_idx]
        temp = pts[:, 1]
        valid_idx = np.where(abs(temp - Center[1]) < Width[1]/2-0.01)
        pts = pts[valid_idx]
        p_colors = p_colors[valid_idx]
        temp = pts[:, 2]
        valid_idx = np.where(abs(temp - Center[2]-0.005) < Width[2]/2-0.015)
        target_pcd.points = o3d.utility.Vector3dVector(pts[valid_idx])
        target_pcd.colors = o3d.utility.Vector3dVector(p_colors[valid_idx])
        
        voxel_size = 0.002
        # target_pcd = crop_point_cloud(target_pcd)
        fitness = [0., 0.]
        final_trans = []
        print(len)
        for _ in range(8):
            fitness = [0., 0.]
            final_trans = []
            for i in range(2):
                source_pcd = source_pcds[i]
                # setVisual(source_pcd)
                # setVisual(target_pcd)
                # print(len(source_pcd.points))
                # print(len(target_pcd.points))
                # print(len(source_pcd.normals))
                # print(len(target_pcd.normals))
                # setVisual(source_pcd)

                source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

                result_global_registration = Global_Registration(source_pcd, target_pcd, voxel_size = voxel_size)
                # print('result_global_registration\n', result_global_registration.transformation)
                print('propertyfitness\n', result_global_registration.fitness)
                fitness[i] += result_global_registration.fitness
                print(len(source_pcd.points))
                print(len(target_pcd.points))
                result_global_registration = execute_refine_Registration(source_pcd, target_pcd, result_global_registration, voxel_size = voxel_size)
                print('propertyfitness\n', result_global_registration.fitness)
                fitness[i] += result_global_registration.fitness

                # result_global_registration = execute_refine_Registration(source_pcd, target_pcd, result_global_registration, voxel_size = voxel_size)
                # print('propertyfitness\n', result_global_registration.fitness)
                # fitness[i] += result_global_registration.fitness
                print(len(source_pcd.points))
                print(len(target_pcd.points))
                current_transformation = result_global_registration.transformation
                result_icp = o3d.registration.registration_colored_icp(
                    source_pcd, target_pcd, voxel_size, result_global_registration.transformation,
                    o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=30))
                print('result_icp\n', result_icp.fitness)
                fitness[i] += result_icp.fitness

                current_transformation = result_icp.transformation
                result_icp = o3d.registration.registration_colored_icp(
                    source_pcd, target_pcd, voxel_size, current_transformation,
                    o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=30))
                print('result_icp\n', result_icp.fitness)
                fitness[i] += result_icp.fitness
                
                # current_transformation = result_icp.transformation
                # result_icp = o3d.registration.registration_colored_icp(
                #     source_pcd, target_pcd, voxel_size, current_transformation,
                #     o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                #                                             relative_rmse=1e-6,
                #                                             max_iteration=30))
                # print('result_icp\n', result_icp.fitness)
                # fitness[i] += result_icp.fitness

                final_trans.append(copy.deepcopy(result_icp.transformation))
                # _, transed_pcd, _ = setTransform(source_pcd, target_pcd, current_transformation)
                # if result_pcd is None:
                #     result_pcd = copy.deepcopy(transed_pcd)
                # else:
                #     result_pcd = result_pcd + transed_pcd
                # transed_pcds.append(transed_pcd)
                # setVisual(result_pcd)
                # o3d.io.write_point_cloud(Data_Dir + "/test2/mdfk_" + str(i) + ".pcd", result_pcd)
            idx = 0 if fitness[0] > fitness[1] else 1
            transformation = final_trans[idx]
            result_pcd, source_transed, _ = setTransform(source_pcds[idx], target_pcd, transformation, True)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            o3d.io.write_point_cloud(Data_Dir + "/result/result_" + current_time + ".pcd", result_pcd)
            setVisual(result_pcd)
            source_transed = source_transed.voxel_down_sample(voxel_size=0.015)
            new_target = delete_point_cloud(target_pcd, source_transed)
            setVisual(new_target)
        

    if Test_GlobalRegistration:
        target_data = Data_Dir + "/cfg_2/mdfk0.pcd"
        source_datas = []
        for i in range(9):
            source_data = Data_Dir + "/cfg_2/mdfk" + str(i+1) + ".pcd"
            source_datas.append(source_data)

        # target_data_2 = Data_Dir + "/blue_light/mdfk2.pcd"
        # metal_bin = Data_Dir + "/part_model/solomon_part_12000.pcd"

        # read pcd or ply data
        target_pcd = o3d.io.read_point_cloud(target_data).voxel_down_sample(voxel_size=0.002)
        source_pcds = []
        for i in range(9):
            source_pcd = o3d.io.read_point_cloud(source_datas[i]).voxel_down_sample(voxel_size=0.002)
            source_pcds.append(source_pcd)
        transed_pcds = []
        result_pcd = None
        #==============Stage 1================
        voxel_size = 0.002
        # setVisual(source_pcd)
        # cl, ind = source_pcd.remove_statistical_outlier(nb_neighbors=500, std_ratio=0.3)
        target_pcd = crop_point_cloud(target_pcd)
        for i in range(9):
            source_pcd = crop_point_cloud(source_pcds[i])
            # setVisual(source_pcd)
            # setVisual(target_pcd)

            result_global_registration = Global_Registration(source_pcd, target_pcd, voxel_size = voxel_size)
            print('propertyfitness\n', result_global_registration.fitness)
            result_global_registration = execute_refine_Registration(source_pcd, target_pcd, result_global_registration, voxel_size = voxel_size)
            print('propertyfitness\n', result_global_registration.fitness)
            result_global_registration = execute_refine_Registration(source_pcd, target_pcd, result_global_registration, voxel_size = voxel_size)
            print('propertyfitness\n', result_global_registration.fitness)
            if result_global_registration.fitness < 0.1:
                continue
            source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            current_transformation = result_global_registration.transformation
            result_icp = o3d.registration.registration_colored_icp(
                source_pcd, target_pcd, voxel_size, result_global_registration.transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=30))
            print('result_icp\n', result_icp.fitness)
            if result_icp.fitness < 0.2:
                continue
            current_transformation = result_icp.transformation
            result_icp = o3d.registration.registration_colored_icp(
                source_pcd, target_pcd, voxel_size, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=30))
            print('result_icp\n', result_icp.fitness)
            if result_icp.fitness < 0.3:
                continue
            current_transformation = result_icp.transformation
            result_icp = o3d.registration.registration_colored_icp(
                source_pcd, target_pcd, voxel_size, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=30))
            print('result_icp\n', result_icp.fitness)
            if result_icp.fitness < 0.4:
                continue
            current_transformation = result_icp.transformation
            _, transed_pcd, _ = setTransform(source_pcd, target_pcd, current_transformation)
            if result_pcd is None:
                result_pcd = copy.deepcopy(transed_pcd)
            else:
                result_pcd = result_pcd + transed_pcd
            transed_pcds.append(transed_pcd)
            setVisual(result_pcd)
            o3d.io.write_point_cloud(Data_Dir + "/test3/mdfk_" + str(i) + ".pcd", result_pcd)

        o3d.io.write_point_cloud(Data_Dir + "/test3/mdfk_mix.pcd", result_pcd)

        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(result_pcd, depth=10)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        pcd = mesh.sample_points_poisson_disk(number_of_points=100000, init_factor=5).voxel_down_sample(voxel_size=0.002)
        o3d.io.write_point_cloud(Data_Dir + "/test3/sampl_from_mesh.pcd", pcd)
        
        # source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # cl, ind = source_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # source_pcd = source_pcd.select_by_index(ind)
        # cl, ind = source_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # source_pcd = source_pcd.select_by_index(ind)
        # cl, ind = source_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # source_pcd = source_pcd.select_by_index(ind)
        # cl, ind = source_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # source_pcd = source_pcd.select_by_index(ind)
        # cl, ind = source_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # source_pcd = source_pcd.select_by_index(ind)
        # cl, ind = source_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # source_pcd = source_pcd.select_by_index(ind)
        # setVisual(source_pcd)
        # cl, ind = target_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # target_pcd = target_pcd.select_by_index(ind)
        # cl, ind = target_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # target_pcd = target_pcd.select_by_index(ind)
        # cl, ind = target_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # target_pcd = target_pcd.select_by_index(ind)
        # cl, ind = target_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # target_pcd = target_pcd.select_by_index(ind)
        # cl, ind = target_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # target_pcd = target_pcd.select_by_index(ind)
        # cl, ind = target_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        # target_pcd = target_pcd.select_by_index(ind)
        # setVisual(target_pcd)

        # setVisual(source_pcd)
        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source_pcd, depth=10)
        # # setVisual(mesh)

        # vertices_to_remove = densities < np.quantile(densities, 0.1)
        # mesh.remove_vertices_by_mask(vertices_to_remove)
        # # setVisual(mesh)
        # # mesh = mesh.filter_smooth_taubin(number_of_iterations=50)
        # mesh = mesh.filter_smooth_simple(number_of_iterations=20)
        # mesh.compute_vertex_normals()
        # # mesh.compute_vertex_normals()
        # setVisual(mesh)

        # triangle_clusters, cluster_n_triangles, cluster_area = (
        # mesh.cluster_connected_triangles())
        # triangle_clusters = np.asarray(triangle_clusters)
        # cluster_n_triangles = np.asarray(cluster_n_triangles)
        # cluster_area = np.asarray(cluster_area)
        # # mesh = copy.deepcopy(mesh)
        # largest_cluster_idx = cluster_n_triangles.argmax()
        # triangles_to_remove = triangle_clusters != largest_cluster_idx
        # mesh.remove_triangles_by_mask(triangles_to_remove)
        # setVisual(mesh)
        # pcd = mesh.sample_points_poisson_disk(number_of_points=30000, init_factor=5)
        # setVisual(pcd)
                                                    
        # print("result_global_registration\n", current_transformation)
        # print("2 ground transformation\n", result_global_registration.transformation)
        # current_transformation = result_global_registration.transformation

        # Stage_1_result_pcd, _, _ = setTransform(source_pcd, target_pcd, current_transformation)
        # print('fuck')
        # setVisual(Stage_1_result_pcd)
        # #==============Stage 1================/
        
        # Stage_1_SegNClustering_result_pcd = SegNClustering(Stage_1_result_pcd, distance_threshold=0.01, eps=2, min_points=10)
        # print('fuck')
        # setVisual(Stage_1_SegNClustering_result_pcd)
        # print('fuck')
        # #==============Stage 2================
        # result_global_registration_2 = Global_Registration(metal_pcd, Stage_1_SegNClustering_result_pcd, voxel_size = voxel_size)
        # print('fuck')
        # print(result_global_registration_2)
        # print("metal to 2 ground transformation ", result_global_registration_2.transformation)
        # print('fuck')
        # Stage_2_result_pcd, _, _ = setTransform(metal_pcd, Stage_1_SegNClustering_result_pcd, result_global_registration_2.transformation, paintUniformColor = True)
        # print('fuck')
        # setVisual(Stage_2_result_pcd + Stage_1_result_pcd)
        #==============Stage 2================/
