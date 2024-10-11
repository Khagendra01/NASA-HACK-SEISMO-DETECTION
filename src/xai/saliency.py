import torch
import numpy as np


class SaliencyMap:

    def __init__(self, model):
        self.model = model

    def compute(self, input_tensor, target_index=None):
        self.model.eval()
        input_tensor.requires_grad_()

        output = self.model(input_tensor)

        if target_index is None:
            target = output.max()
        else:
            target = output[:, target_index]

        self.model.zero_grad()
        target.backward()

        gradients = input_tensor.grad.cpu().numpy()
        saliency = np.abs(gradients).squeeze()

        saliency -= saliency.min()
        saliency /= saliency.max() + 1e-8

        return saliency