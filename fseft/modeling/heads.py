
import torch


class LinearProbe(torch.nn.Module):
    def __init__(self, feature_size=48, out_channels=1, zs_weights=None):
        super().__init__()
        self.feature_size = feature_size

        # Set new classifier
        self.linear = torch.nn.Conv3d(feature_size, out_channels, kernel_size=1, stride=1, padding=0)
        if zs_weights is not None:
            self.linear.weight = torch.nn.Parameter(zs_weights["weights"].clone())
            self.linear.bias = torch.nn.Parameter(zs_weights["bias"].clone())

    def forward(self, x):
        out = self.linear(x)

        return out


def retrieve_zeroshot_weights(args, model):

    # Set new classification head (init with zero-shot weights)
    if (args.model_id in ["fseft", "btcv"] and args.method != "scratch") and args.universal_indexes is not None:
        weights, biases = [], []
        # Add weights for background class in multi-class scenario.
        if args.objective == "multiclass":
            weights.append(torch.nn.init.xavier_uniform_(model.classifier.weight[0].data.unsqueeze(0).clone()))
            biases.append(torch.zeros_like(model.classifier.bias[0].data.unsqueeze(0).clone()))
        for i_organ in args.universal_indexes:
            # Select index for the target class from the pre-trained model
            idx = i_organ - 1
            if args.model_id == "btcv":
                idx += 1
            # Retrieve weights and biases
            weights.append(model.classifier.weight[idx].data.unsqueeze(0))
            biases.append(model.classifier.bias[idx].data.unsqueeze(0))

        zs_weights = {"weights": torch.concat(weights, 0), "bias": torch.concat(biases, 0)}
        print("Init classifier with generalization weights!")
    else:
        zs_weights = None

    return zs_weights
