import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet18(nn.Module):
    def __init__(self, base_model):
        super(ModifiedResNet18, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)  # Raw logits from the model
        return logits  # No softmax applied

def create_model():
    # Load pre-trained ResNet18
    base_model = models.resnet18(pretrained=True)
    num_ftrs = base_model.fc.in_features

    # Update the final layer for binary classification
    base_model.fc = nn.Linear(num_ftrs, 2)  # Output raw logits
    return ModifiedResNet18(base_model)

