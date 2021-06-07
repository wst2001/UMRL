import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg, vgg16



class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.model = vgg16(pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.features = []
        feature_layers = [3, 8, 15, 22]
        for l in feature_layers:
            layer = self.model.features[l]
            layer.register_forward_hook(self.hook_function)
        

    def hook_function(self, module, inputs, outputs):
        self.features.append(outputs)

    def forward(self, inputs):
        self.features.clear()
        self.model.eval()
        outputs = self.model(inputs)
        return [*self.features]