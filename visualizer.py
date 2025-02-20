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
import torch
import pytorch3d
from pytorch3d.structures import Meshes
import pytorch3d.renderer as rdr
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Pointclouds, Meshes
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
device = torch.device('cuda')
from utils_vox import voxelize_xyz


# Point Cloud Rendering
def render_point_cloud(points, output_path, num_views=120, image_size=256):
    if points.dim() == 2:
        points = points.unsqueeze(0)  # Ensure shape (B, N, 3)
    colors = torch.tensor([[1.0, 0.0, 0.0]] * points.shape[1], device=device).unsqueeze(0)  # Default red color
    angles = torch.linspace(-180, 180, num_views)
    R, T = rdr.look_at_view_transform(dist=5, elev=0, azim=angles)
    cameras = rdr.FoVPerspectiveCameras(R=R, T=T, device=device)
    lights = rdr.PointLights(location=[[0, 0, -3]], device=device)
    
    rasterizer = rdr.PointsRasterizer(cameras=cameras)
    compositor = rdr.AlphaCompositor()
    renderer = rdr.PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    
    pc = Pointclouds(points=points, features=colors).to(device)
    images = renderer(pc.extend(num_views), cameras=cameras, lights=lights)
    images = images.detach().cpu().numpy()[..., :3] * 255
    imageio.mimsave(output_path, images.astype(np.uint8), fps=30, loop=0)

# Mesh Rendering
def render_mesh(mesh, output_path, num_views=120, image_size=512, distance=2.7, elevation=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = mesh.to(device)
    verts = mesh.verts_packed()
    verts_rgb = torch.ones_like(verts)[None]  # Default white color
    mesh.textures = rdr.TexturesVertex(verts_features=verts_rgb)
    
    rasterizer = rdr.MeshRasterizer(
        cameras=rdr.FoVPerspectiveCameras(device=device),
        raster_settings=rdr.RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1
        )
    )
    shader = rdr.HardPhongShader(device=device, cameras=rdr.FoVPerspectiveCameras(device=device))
    renderer = rdr.MeshRenderer(rasterizer=rasterizer, shader=shader)
    angles = torch.linspace(-180, 180, num_views)
    images = []
    
    for angle in tqdm(angles):
        R, T = rdr.look_at_view_transform(dist=distance, elev=elevation, azim=angle)
        cameras = rdr.FoVPerspectiveCameras(R=R, T=T, device=device)
        
        torch.cuda.synchronize()
        render = renderer(mesh, cameras=cameras)
        images.append((render[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8))
    
    imageio.mimsave(output_path, images, fps=30)

# Voxel Rendering
def render_voxels(voxels, output_path, num_views=120):
    max_val = 1.0
    min_val = 0.0
    data = voxels.detach().cpu().numpy()
    data = np.squeeze(data)
    size = voxels.shape[0]
    vertices, faces = mcubes.marching_cubes(
        mcubes.smooth(data),
        isovalue=0.5
    )
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    vertices = (vertices/size)*(max_val-min_val)+min_val

    colors = torch.ones_like(vertices)
    textures = pytorch3d.renderer.TexturesVertex(colors.unsqueeze(0))
    mesh = Meshes([vertices],[faces],textures=textures).to(device)
    # mesh = pytorch3d.ops.cubify(voxels, thresh=threshold).to(device)
    
    # points = torch.clamp(points, 0, 1)
    # voxels = voxelize_xyz(points.unsqueeze(0), Z, Y, X)
    # voxels = voxels.squeeze().cpu().numpy()
    # verts, faces = mcubes.marching_cubes(voxels, 0.5)
    # mesh = Meshes(verts=[torch.tensor(verts, device=device)], faces=[torch.tensor(faces, device=device)])
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