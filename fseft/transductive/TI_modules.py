import torch
import numpy as np

from fseft.transductive.utils import preprocess_query, store_features, estimate_target

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProportionConstraint(object):
    def __init__(self, args, model):
        self.weight = args.lambda_TI
        self.margin = args.margin
        self.scale_feature_volume = 2  # Scaling stored features from backbone due to memory constraints

        # Store features (for time saving) from backbone (only valid if backbone is frozen)
        self.x_query_batched = preprocess_query(args)
        self.query_size = np.prod(self.x_query_batched.shape)/(self.scale_feature_volume**3)
        self.feats_query = store_features(self.x_query_batched, model, scale=self.scale_feature_volume)

        # Estimate target size
        self.ptarget = (estimate_target(args)/(self.scale_feature_volume**3)) / self.query_size
        self.pReal = (np.sum(args.query["label"])/(self.scale_feature_volume**3)) / self.query_size

    def step(self, model, optimizer):
        model.train()

        # Apply classifier over features and store target size
        phat = torch.tensor(0.).to(device)
        for i in range(len(self.feats_query)):
                phat += torch.sum(torch.sigmoid(
                    model.module.classifier(torch.tensor(self.feats_query[i]).to(device).to(torch.float32))))
        phat /= self.query_size
        print("Proportion target: " + str(self.ptarget) + " || " + "Proportion pred: " + str(phat.item()) +
              " || " + "Proportion real: " + str(self.pReal))

        # Normalize error w.r.t target size (due to small relative size of organs)
        # relative_error = (self.ptarget - phat) / self.ptarget
        relative_error = (self.ptarget - phat) / max(phat.item(), self.ptarget)

        # ReLU-based L1 penalty with margin \__/
        if relative_error.item() > self.margin:
            penalty = torch.abs(relative_error) - self.margin
        elif relative_error.item() < -self.margin:
            penalty = torch.abs(relative_error) + self.margin
        else:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            penalty = torch.tensor(0.)
            return penalty.item()

        # Weight
        weighted_penalty = self.weight*penalty
        # Backward
        weighted_penalty.backward()
        # Step
        optimizer.step()

        torch.cuda.empty_cache()
        return penalty.item()