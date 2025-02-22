from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        self.activations = {}
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            
            self.encoder[0].register_forward_hook(self._get_activation('conv1'))  # First conv layer
            self.encoder[4].register_forward_hook(self._get_activation('layer1')) # ResNet layer1
            self.encoder[5].register_forward_hook(self._get_activation('layer2')) # ResNet layer2
            self.encoder[6].register_forward_hook(self._get_activation('layer3')) # ResNet layer3
            self.encoder[7].register_forward_hook(self._get_activation('layer4')) # ResNet layer4
            
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # pass
            # TODO:
            self.decoder = torch.nn.Sequential(
                nn.Linear(512, 2048)
            )
            self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 512, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
            )
            self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
            )
            self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
            )
            self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
            )
            self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
            )
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, self.n_point),
                nn.LeakyReLU(),
                nn.Linear(self.n_point, self.n_point * 3),
                nn.Tanh()
            )            
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            num_vertices = mesh_pred.verts_packed().shape[0]
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, num_vertices * 3),
            )
            
    def _get_activation(self, name):
        """Hook function to store activations"""
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def get_intermediate_features(self):
        """Return stored activations"""
        return self.activations
    

    def forward(self, images, args):
        results = dict()
        self.activations = {}
        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
                voxel_pred = self.decoder(encoded_feat)
                voxel_pred = voxel_pred.view((-1, 256, 2, 2, 2))
                voxel_pred = self.layer1(voxel_pred)
                voxel_pred = self.layer2(voxel_pred)
                voxel_pred = self.layer3(voxel_pred)
                voxel_pred = self.layer4(voxel_pred)
                voxel_pred = self.layer5(voxel_pred)
                return voxel_pred

        elif args.type == "point":
            # TODO:
            point_cloud_flat = self.decoder(encoded_feat)
            pointclouds_pred = point_cloud_flat.view(-1, self.n_point, 3)           
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)          
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

    def visualize_features(self, output_dir):
        """
        Visualize intermediate features
        Args:
            output_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, feat in self.activations.items():
            # Take first sample in batch
            feature_map = feat[0].mean(dim=0)  # Average across channels
            
            plt.figure(figsize=(10, 10))
            plt.imshow(feature_map.detach().cpu(), cmap='viridis')
            plt.title(f'Feature Map: {name}')
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, f'feature_map_{name}.png'))
            plt.close()