import torch
import numpy as np

from monai.networks.blocks.dynunet_block import UnetResBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SpatialAdapter(torch.nn.Module):
    def __init__(self, adapter_kernel_size=3, feature_size=48, spatial_dims=3, norm_name="instance"):
        super().__init__()
        self.feature_size = feature_size
        self.adapter = UnetResBlock(spatial_dims, feature_size, feature_size, kernel_size=adapter_kernel_size,
                                    stride=1, norm_name=norm_name)

    def forward(self, x):
        out = self.adapter(x)

        return out


class LinearProbe(torch.nn.Module):
    def __init__(self, feature_size=48, out_channels=1):
        super().__init__()
        self.feature_size = feature_size
        self.linear = torch.nn.Conv3d(feature_size, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.linear(x)

        return out


class Prototype(torch.nn.Module):
    def __init__(self, feature_size=48, logit_scale_init_value=0.07):
        super().__init__()
        self.feature_size = feature_size
        self.temperature = torch.nn.Parameter(torch.tensor(1 / logit_scale_init_value))
        self.prototype = torch.nn.Parameter(torch.randn((feature_size, 1)))

    def forward(self, out):

        # Reshape vision embedding
        out = out.permute((0, 2, 3, 4, 1))

        # Compute cosine similarity
        cossim = out.matmul(self.prototype)
        norm_prototype = self.prototype.norm(dim=0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        norm_out = out.norm(dim=-1).unsqueeze(-1).repeat((1, 1, 1, 1, self.prototype.shape[-1]))
        norm = (norm_prototype * norm_out) + 1e-10
        cossim /= norm
        cossim = cossim.permute((0, -1, 1, 2, 3))

        return self.temperature * cossim

    def init_prototype(self, model, train_loader, method):
        print("Initializing prototype-based classification... ", end="\n")
        model.eval()

        # Get foreground sample
        positives = 0
        while positives == 0:
            sample = train_loader.dataset.__getitem__(0)
            positives = np.max(sample[0]["label"].numpy())

        # Forward features
        model.out_features = True  # Temporarily to get backbone features
        with torch.no_grad():
            feats = model(sample[0]["image"].unsqueeze(0).to(device).to(torch.float32))
            if method == 'spatial_adapter':
                feats = model.classifier[0](feats)
        model.out_features = False  # Back to normal

        # Compute prototype for foreground
        mask = (sample[0]["label"] == 1).to(device).to(torch.float32)
        # Average features on foreground region
        prototype = (torch.sum(feats * mask, (2, 3, 4)) / torch.sum(sample[0]["label"])).cpu().detach()

        # Set initialized trainable prototype parameter
        self.prototype = torch.nn.Parameter(torch.tensor(prototype.clone().transpose(0, 1).numpy()).to(device))
        model.train()
