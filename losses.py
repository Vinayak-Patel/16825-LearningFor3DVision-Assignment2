import torch
import pytorch3d

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# implement some loss for binary voxel grid
    lossfn = torch.nn.BCEWithLogitsLoss()
    voxel_src = voxel_src.view(-1)
    voxel_tgt = voxel_tgt.view(-1)
    loss = lossfn(voxel_src, voxel_tgt)
    return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
    dist_st = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1)
    dist_ts = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1)
    loss_chamfer = torch.mean(dist_st) + torch.mean(dist_ts)
    return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
    loss_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(mesh_src)
    return loss_laplacian