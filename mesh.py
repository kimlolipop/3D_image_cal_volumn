import open3d as o3d
import matplotlib.pyplot as plt
# from pyntcloud import PyntCloud
# from pyntcloud.geometry.models.sphere import create_sphere
# import pyvista as pv
import os

def create_point_clouds(voxel_size, rgb_path, depth_path):
    pcds = []
    for i in range(len(rgb_path)):
        color_raw = o3d.io.read_image('data/rgb/' + rgb_path[i])
        depth_raw = o3d.io.read_image('data/depth/' + depth_path[i])


        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))



        # pcd = o3d.io.read_point_cloud(pathh[i])
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        # pcd_down = pcd_down.compute_vertex_normals()
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcds.append(pcd_down)

    return pcds



# ===============show 3D render==============
color_raw = o3d.io.read_image("data/rgb/02589.jpg")
depth_raw = o3d.io.read_image("data/depth/02589.png")
voxel_size = 0.02

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])



# ======================point cloud==================
rgb_img = os.listdir('data/rgb')
depth_img = os.listdir('data/depth')

print(rgb_img, depth_img)
voxel_size = 0.02
pcds_down = create_point_clouds(voxel_size, rgb_img, depth_img)
o3d.visualization.draw_geometries(pcds_down,)



# =======================Create mesh=================
alpha = 0.5
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# o3d.io.write_triangle_mesh("meshh.ply", mesh) #weite mesh file

