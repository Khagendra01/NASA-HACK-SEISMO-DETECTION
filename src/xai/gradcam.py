import torch
import torch.nn.functional as F
import numpy as np
import scipy.signal as sg


class GradCAM1D:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, target_index=None):

        self.model.eval()

        output = self.model(x)

        if target_index is None:
            target = output.max()
        else:
            target = output[:, target_index]

        self.model.zero_grad()
        target.backward(retain_graph=True)

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=-1)

        cam = torch.zeros(activations.shape[1]).to(x.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.cpu().numpy()


class SeismicGradCAM:

    def __init__(self, model, target_layer):

        self.gradcam = GradCAM1D(model, target_layer)

    def compute(self, input_tensor, original_length):

        cam = self.gradcam.generate(input_tensor)

        cam_resized = sg.resample(cam, original_length)

        cam_resized -= cam_resized.min()
        cam_resized /= cam_resized.max() + 1e-8

        return cam_resized