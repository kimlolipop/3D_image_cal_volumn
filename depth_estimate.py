import torch
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import os

# Depth estimate=========================================
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform

# img = cv2.imread('data/rgb/02589.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# input_batch = transform(img).to(device)

# with torch.no_grad():
#     prediction = midas(input_batch)

#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

# output = prediction.cpu().numpy()
# output.shape
# cv2.imwrite('output.png', output)

# plt.figure()
# plt.imshow(output)
# plt.show()





# point cloud =====================================================
def create_point_clouds(voxel_size, rgb_path, depth_path):
    pcds = []
    for i in range(len(rgb_path)):
        color_raw = o3d.io.read_image('data/rgb/' + rgb_path[i])
        depth_raw = o3d.io.read_image(depth_path)


        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))




        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcds.append(pcd_down)

        # visualize
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([pcd])

    return pcds

rgb_img = os.listdir('data/rgb')
depth_img = 'output.JPG'

print(rgb_img, depth_img)
voxel_size = 0.02
pcds_down = create_point_clouds(voxel_size, rgb_img, depth_img)
o3d.visualization.draw_geometries(pcds_down,)
