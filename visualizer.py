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
        faces_per_pixel=1
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
def render_mesh(mesh, output_path, num_views=120, image_size=512, distance=2.7, elevation=30, textures=None, fov=60):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mesh = mesh.to(device)
    # verts = mesh.verts_packed()
    # verts_rgb = torch.ones_like(verts)[None]  # Default white color
    # mesh.textures = rdr.TexturesVertex(verts_features=verts_rgb)
    
    # rasterizer = rdr.MeshRasterizer(
    #     cameras=rdr.FoVPerspectiveCameras(device=device),
    #     raster_settings=rdr.RasterizationSettings(
    #         image_size=512, blur_radius=0.0, faces_per_pixel=1
    #     )
    # )
    # shader = rdr.HardPhongShader(device=device, cameras=rdr.FoVPerspectiveCameras(device=device))
    # renderer = rdr.MeshRenderer(rasterizer=rasterizer, shader=shader)
    # angles = torch.linspace(-180, 180, num_views)
    # images = []
    
    # for angle in tqdm(angles):
    #     R, T = rdr.look_at_view_transform(dist=distance, elev=elevation, azim=angle)
    #     cameras = rdr.FoVPerspectiveCameras(R=R, T=T, device=device)
        
    #     torch.cuda.synchronize()
    #     render = renderer(mesh, cameras=cameras)
    #     images.append((render[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8))
    
    # imageio.mimsave(output_path, images, fps=30)
    # print(device)
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)     # (N_f, 3) -> (1, N_f, 3)
    
   
    if textures is None:
        # textures = torch.ones_like(vertices)  # (1, N_v, 3)
        # textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
        
        if vertices.numel() > 0:
            textures = torch.ones_like(vertices)
            textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
        else:
            textures = torch.ones_like(vertices)  # Default to zero textures for empty tensors

        # if vertices.numel() > 0:
        #     textures = torch.ones_like(vertices)
        # else:
        #     print("Vertices are empty; using default zero textures.")
        #     textures = torch.zeros(1, 1, 1)
    
    render_mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
    ).to(device)
    
    azimuth = np.linspace(-180, 180, num=num_views)
    R, T = pytorch3d.renderer.look_at_view_transform(dist = distance, elev = elevation, 
                                                     azim =azimuth)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov= fov, device=device)
    renderer = utils.get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, 3]], device=device)
    
    images = renderer(render_mesh.extend(num_views), cameras= cameras, lights= lights)
    images = images.detach().cpu().numpy()[..., :3]
    images = (images * 255).clip(0, 255).astype(np.uint8)
    # images = images.cpu().detach().numpy()
    imageio.mimsave(output_path, images, fps=30, format='gif', loop=0)
    return

# Voxel Rendering
def render_voxels(voxels, output_path, num_views=120):
    # max_val = 1.0
    # min_val = 0.0
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Ensure voxels are within valid range
    # voxels = voxels.clamp(0, 1)
    
    # print("Voxel min:", voxels.min().item(), "Voxel max:", voxels.max().item())

    # data = voxels.detach().cpu().numpy()
    # data = np.squeeze(data)

    # size = voxels.shape[0]
    
    # # Smooth and apply marching cubes
    # vertices, faces = mcubes.marching_cubes(mcubes.smooth(data), isovalue=0.5)

    # # Convert to PyTorch tensors and ensure correct dtype
    # vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    # faces = torch.tensor(faces.astype(np.int64), device=device)
    # vertices = vertices.unsqueeze(0)
    # faces = faces.unsqueeze(0)
    # # Normalize vertices
    # vertices = (vertices / size) * (max_val - min_val) + min_val

    # # Ensure correct shape for textures
    # colors = torch.ones_like(vertices, device=device)  # (1, V, 3)
    # textures = pytorch3d.renderer.TexturesVertex(verts_features=colors)

    # mesh = Meshes([vertices], [faces], textures=textures).to(device)

    # # Mesh Renderer
    # renderer = get_mesh_renderer()

    # num_steps = num_views
    # radius = 2.0
    # images = []
    
    # for i in range(num_steps):
    #     angle = 2 * torch.pi * i / num_steps
    #     angle = torch.tensor(angle, device=device)
    #     camera_position = torch.tensor([[radius * torch.cos(angle), 
    #                                      0, 
    #                                      radius * torch.sin(angle)]], device=device)
        
    #     at = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    #     up = torch.tensor([[0.0, 1.0, 0.0]], device=device)

    #     R, T = rdr.look_at_view_transform(eye=camera_position, at=at, up=up)
    #     cameras = rdr.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    #     lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        
    #     print("checkpoint1")
    #     rend = renderer(mesh, cameras=cameras, lights=lights)
    #     print("checkpoint2")
    #     rend = rend.detach().cpu().numpy()[0, ..., :3]
    #     print("checkpoint3")
    #     images.append((rend * 255).clip(0, 255).astype(np.uint8))
    
    # duration = 1000 // 15
    # imageio.mimsave(output_path, images, duration=duration, loop=0)
    mesh = pytorch3d.ops.cubify(voxels, thresh=0.5).to(device)
    
    # # Check if the mesh is empty
    # if len(mesh.verts_list()[0]) == 0:
    #     print("Generated mesh is empty. Skipping visualization.")
    #     return
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