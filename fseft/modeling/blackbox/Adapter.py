import torch


def set_peft(model, args):
    # Freeze params
    freeze_params(model)


def set_bb(args):
    modules = []

    # Set backbone head
    if args.method == 'bb3DAdapter':
        modules.append(SpatialAdapter(adapter_kernel_size=3, feature_size=args.model_cfg["fout"], residual=True))
    if args.method == 'bbAdapter':
        modules.append(SpatialAdapter(adapter_kernel_size=1, feature_size=args.model_cfg["fout"], residual=True))

    return modules


def set_training_mode(model, args):
    model.eval()


def freeze_params(model):
    print("Freezing weights... ", end="\n")
    for name, param in model.named_parameters():
        param.requires_grad = False


class SpatialAdapter(torch.nn.Module):
    def __init__(self, feature_size=48, residual=False, adapter_kernel_size=3):
        super().__init__()
        self.feature_size = feature_size
        self.residual = residual
        self.alpha_residual = 1

        self.adapter = torch.nn.Sequential(
            torch.nn.Conv3d(feature_size, feature_size, kernel_size=adapter_kernel_size, stride=1, padding="same",
                            bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv3d(feature_size, feature_size, kernel_size=adapter_kernel_size, stride=1, padding="same",
                            bias=False),
        )

        # Init B matrix with 0s
        torch.nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, x):
        out = self.adapter(x)

        if self.residual:
            out = x + out * self.alpha_residual

        return out