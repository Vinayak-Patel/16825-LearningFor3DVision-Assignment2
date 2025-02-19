"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import pytorch3d
import torch
from pytorch3d.structures import Meshes
import pytorch3d.renderer as rdr
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Pointclouds, Meshes
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm.notebook import tqdm
device = torch.device('cuda')
from utils_vox import voxelize_xyz

# ## Mesh Start
# def get_mesh_renderer(image_size=512):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     R, T = rdr.look_at_view_transform(2.7, 0, 0)
#     cameras = rdr.FoVPerspectiveCameras(device=device, R=R, T=T)
    
#     raster_set = rdr.RasterizationSettings(
#         image_size=image_size, 
#         blur_radius=0.0, 
#         faces_per_pixel=1
#     )
    
#     lights = rdr.PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    
#     render = rdr.MeshRenderer(
#         rasterizer=rdr.MeshRasterizer(
#             cameras=cameras, 
#             raster_settings=raster_set
#         ),
#         shader=rdr.HardPhongShader(
#             device=device, 
#             cameras=cameras,
#             lights=lights
#         )
#     )
#     return render

# def render_360_degree_mesh(mesh, device, image_size=512, num_views=72, distance=2.75, elevation=30):
#     renderer = get_mesh_renderer(image_size=image_size)
#     angles = torch.linspace(-180, 180, num_views)
#     lights = rdr.PointLights(location=[[0, 0, -3]], device=device)
#     images = []
    
#     for angle in tqdm(angles):
#         R, T = rdr.look_at_view_transform(dist=distance, elev=elevation, azim=angle)
#         cameras = rdr.FoVPerspectiveCameras(R=R, T=T, device=device)
        
#         render = renderer(mesh, cameras=cameras, lights=lights)
#         image = render[0, ..., :3].cpu().numpy()
#         image = (image * 255).astype(np.uint8)
#         images.append(image)
    
#     return images
   
# def save_gif(images, output_path, fps=24):
#     duration = 1000 // fps
#     imageio.mimsave(
#         output_path,
#         images,
#         duration=duration,
#         loop=0
#     )
    
# obj_filename = "../data/cow.obj"
# mesh = load_objs_as_meshes([obj_filename], device=device)

# render_img = render_360_degree_mesh(
#     mesh,
#     device=device,
#     image_size=512,
#     num_views=120,
#     distance=2.7,
#     elevation=30
# )

# save_gif(render_img, 'mesh_360_hardphongshader.gif', fps=30)
# ## Mesh End
# #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Point Cloud Start
# def load_rgbd_data(path="data/rgbd_data.pkl"):
#     with open(path, "rb") as f:
#         data = pickle.load(f)
#     return data

# data = load_rgbd_data(path="../data/rgbd_data.pkl")

# def unproject_depth_image(image, mask, depth, camera):
#     """
#     Unprojects a depth image into a 3D point cloud.

#     Args:
#         image (torch.Tensor): A square image to unproject (S, S, 3).
#         mask (torch.Tensor): A binary mask for the image (S, S).
#         depth (torch.Tensor): The depth map of the image (S, S).
#         camera: The Pytorch3D camera to render the image.
    
#     Returns:
#         points (torch.Tensor): The 3D points of the unprojected image (N, 3).
#         rgba (torch.Tensor): The rgba color values corresponding to the unprojected
#             points (N, 4).
#     """
#     device = camera.device
#     assert image.shape[0] == image.shape[1], "Image must be square."
#     image_shape = image.shape[0]
#     ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
#     Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
#     xy_depth = torch.dstack([X, Y, depth])
#     points = camera.unproject_points(
#         xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
#     )
#     points = points[mask > 0.5]
#     rgb = image[mask > 0.5]
#     rgb = rgb.to(device)

#     # For some reason, the Pytorch3D compositor does not apply a background color
#     # unless the pointcloud is RGBA.
#     alpha = torch.ones_like(rgb)[..., :1]
#     rgb = torch.cat([rgb, alpha], dim=1)

#     return points, rgb

# points1, colors1 = unproject_depth_image(torch.tensor(data["rgb1"]), 
#                                               torch.tensor(data["mask1"]),
#                                               torch.tensor(data["depth1"]),
#                                               data["cameras1"])

# pc1 = pytorch3d.structures.Pointclouds(points=points1.unsqueeze(0), features=colors1.unsqueeze(0)).to(device)
# num_views=120
# angles = torch.linspace(-180, 180, num_views)
# image_size = 256

# R, T = rdr.look_at_view_transform(dist = 10, elev = 0, azim = angles)

# R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()@R

# cameras = rdr.FoVPerspectiveCameras(R=R, T=T, device=device)
# lights = rdr.PointLights(location=[[0, 0, -3]], device=device)
# renderer = utils.get_points_renderer(image_size=image_size, device=device)

# images = renderer(pc1.extend(num_views), cameras = cameras, lights = lights)
# images = images.cpu().numpy()[..., :3] 
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("pc1.gif", images, fps=30, loop = 0)

# points2, colors2 = utils.unproject_depth_image(torch.tensor(data["rgb2"]), 
#                                               torch.tensor(data["mask2"]),
#                                               torch.tensor(data["depth2"]),
#                                               data["cameras2"])

# pc2 = pytorch3d.structures.Pointclouds(points=points2.unsqueeze(0), features=colors2.unsqueeze(0)).to(device)

# images = renderer(pc2.extend(num_views), cameras = cameras, lights = lights)
# images = images.cpu().numpy()[..., :3] 
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("pc2.gif", images, fps=30, loop = 0)

# pc3 = pytorch3d.structures.Pointclouds(points=torch.cat((points1,points2), 0).unsqueeze(0), features=torch.cat((colors1,colors2), 0).unsqueeze(0),).to(device)

# images = renderer(pc3.extend(num_views), cameras= cameras, lights= lights)
# images = images.cpu().numpy()[..., :3] 
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("pc3.gif", images, fps=30, loop = 0)

# ## Point Cloud End
# #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Voxel Start


# ## Voxel End


# Point Cloud Rendering
def render_point_cloud(points, output_path, num_views=120, image_size=256):
    if points.dim() == 2:
        points = points.unsqueeze(0)  # Ensure shape (B, N, 3)
    colors = torch.tensor([[1.0, 0.0, 0.0]] * points.shape[1], device=device).unsqueeze(0)  # Default red color
    angles = torch.linspace(-180, 180, num_views)
    R, T = rdr.look_at_view_transform(dist=10, elev=0, azim=angles)
    cameras = rdr.FoVPerspectiveCameras(R=R, T=T, device=device)
    lights = rdr.PointLights(location=[[0, 0, -3]], device=device)
    
    rasterizer = rdr.PointsRasterizer(cameras=cameras)
    compositor = rdr.AlphaCompositor()
    renderer = rdr.PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    
    pc = Pointclouds(points=points, features=colors).to(device)
    images = renderer(pc.extend(num_views), cameras=cameras, lights=lights)
    images = images.cpu().numpy()[..., :3] * 255
    imageio.mimsave(output_path, images.astype(np.uint8), fps=30, loop=0)

# Mesh Rendering
def render_mesh(mesh, output_path, num_views=120, image_size=512, distance=2.7, elevation=30):
    renderer = rdr.MeshRenderer()
    angles = torch.linspace(-180, 180, num_views)
    images = []
    
    for angle in tqdm(angles):
        R, T = rdr.look_at_view_transform(dist=distance, elev=elevation, azim=angle)
        cameras = rdr.FoVPerspectiveCameras(R=R, T=T, device=device)
        render = renderer(mesh, cameras=cameras)
        images.append((render[0, ..., :3].cpu().numpy() * 255).astype(np.uint8))
    
    imageio.mimsave(output_path, images, fps=30)

# Voxel Rendering
def render_voxels(points, output_path, Z=32, Y=32, X=32, num_views=120):
    voxels = voxelize_xyz(points.unsqueeze(0), Z, Y, X)
    voxels = voxels.squeeze().cpu().numpy()
    verts, faces = mcubes.marching_cubes(voxels, 0.5)
    mesh = Meshes(verts=[torch.tensor(verts, device=device)], faces=[torch.tensor(faces, device=device)])
    render_mesh(mesh, output_path, num_views=num_views)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, required=True, choices=["point_cloud", "mesh", "voxel"])
    args = parser.parse_args()
    
    if args.render == "point_cloud":
        data = pickle.load(open("../data/rgbd_data.pkl", "rb"))
        points = torch.tensor(data["points"])
        render_point_cloud(points, "point_cloud.gif")
    
    elif args.render == "mesh":
        mesh = load_objs_as_meshes(["../data/cow.obj"], device=device)
        render_mesh(mesh, "mesh.gif")
    
    elif args.render == "voxel":
        data = pickle.load(open("../data/rgbd_data.pkl", "rb"))
        points = torch.tensor(data["points"])
        render_voxels(points, "voxel.gif")
