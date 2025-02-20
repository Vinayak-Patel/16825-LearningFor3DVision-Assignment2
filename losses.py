import torch

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# implement some loss for binary voxel grid
    lossfn = torch.nn.BCELoss()
    voxel_src = voxel_src.view(-1)
    voxel_tgt = voxel_tgt.view(-1)
    loss = lossfn(voxel_src, voxel_tgt)
    return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
    dist_st = torch.min(torch.sum((point_cloud_src.unsqueeze(2) - point_cloud_tgt.unsqueeze(1))**2, dim=3), dim=2)[0]
    dist_ts = torch.min(torch.sum((point_cloud_tgt.unsqueeze(2) - point_cloud_src.unsqueeze(1))**2, dim=3), dim=2)[0]
    loss_chamfer = torch.mean(dist_st) + torch.mean(dist_ts)
    return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
    vertices = mesh_src.verts_packed()
    edges = mesh_src.edges_packed()
    v1 = vertices[edges[:, 0]]
    v2 = vertices[edges[:, 1]]
    laplacian = v1 - v2
    loss_laplacian = torch.mean(torch.sum(laplacian ** 2, dim=1))
    return loss_laplacian