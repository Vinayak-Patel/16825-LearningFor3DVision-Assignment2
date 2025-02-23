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
import utils
device = torch.device('cuda')

def get_mesh_renderer(image_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    R, T = rdr.look_at_view_transform(2.7, 0, 0)
    cameras = rdr.FoVPerspectiveCameras(device=device, R=R, T=T)
    
    raster_set = rdr.RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1,
        max_faces_per_bin=100000, 
        bin_size=None
    )
    
    lights = rdr.PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    
    render = rdr.MeshRenderer(
        rasterizer=rdr.MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_set
        ),
        shader=rdr.HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    return render

# Point Cloud Rendering
def render_point_cloud(points, output_path, num_views=120, image_size=256):
    if points.dim() == 2:
        points = points.unsqueeze(0)  # Ensure shape (B, N, 3)
    colors = torch.tensor([[1.0, 0.0, 0.0]] * points.shape[1], device=device).unsqueeze(0)  # Default red color
    angles = torch.linspace(-180, 180, num_views)
    R, T = rdr.look_at_view_transform(dist=2.7, elev=0, azim=angles)
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
def render_mesh(mesh, output_path, num_views=30, image_size=256, distance=2.7, textures=None, fov=60, fps=10, elev=1):
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    vertices = vertices.to(device)
    faces = faces.to(device)
    
    # Add batch dimension if needed
    if len(vertices.shape) == 2:
        vertices = vertices.unsqueeze(0)
    if len(faces.shape) == 2:
        faces = faces.unsqueeze(0)
    
        # Create simple vertex colors (normalized position-based coloring)
    if textures is None:
        verts_rgb = torch.ones_like(vertices[0])  # Remove batch dim for calculation
        verts_normalized = (vertices[0] - vertices[0].min()) / (vertices[0].max() - vertices[0].min())
        verts_rgb = verts_normalized
        verts_rgb = verts_rgb.unsqueeze(0)  # Add batch dimension back
        textures = pytorch3d.renderer.TexturesVertex(verts_rgb.to(device))
    
    # Create mesh with textures
    meshes = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=textures
    ).to(device)
    
    # Setup camera parameters
    azim = torch.linspace(-180, 180, num_views)
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=distance,
        elev=torch.ones(num_views) * elev,
        azim=azim
    )
    
    # Create cameras
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        fov=fov,
        device=device
    )
    
    # Create renderer with updated settings
    renderer = get_mesh_renderer(image_size=image_size)
    
    # Render
    meshes_batch = meshes.extend(num_views)
    with torch.no_grad():
        images = renderer(meshes_batch, cameras=cameras)
    
    # Convert images to numpy and save
    images = images.cpu().numpy()
    images = (images[..., :3] * 255).astype(np.uint8)
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as gif
    imageio.mimsave(output_path, images, fps=fps, loop=0)
    return
    

# Voxel Rendering
def render_voxels(voxels, output_path, num_views=120):
    voxels = voxels.squeeze(1)
    mesh = pytorch3d.ops.cubify(voxels, thresh=0.3).to(device)
    
    render_mesh(mesh, output_path,textures=None,num_views= 120, 
                    image_size=256, distance= 3, fov=60, fps=12, elev=1)

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