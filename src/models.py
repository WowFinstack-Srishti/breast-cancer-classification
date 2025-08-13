import torch
import torch.nn as nn
import torchvision.models as models
import timm

class ResNet50Fine(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.net = models.resnet50(pretrained=pretrained)
        in_f = self.net.fc.in_features
        self.net.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.net(x)
    
class ViTModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=2, pretrained=True):
        super().__init__()
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)