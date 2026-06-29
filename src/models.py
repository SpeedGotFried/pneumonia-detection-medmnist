import torch
import torch.nn as nn
import torchvision.models as models

class VanillaCNN(nn.Module):
    """
    A simple custom CNN architecture built from scratch.
    It takes a 3-channel image (resized grayscale) and passes it through
    multiple convolution, batch norm, activation, and max pooling blocks.
    It ends with a Global Average Pooling layer and a linear classifier.
    This design makes it compatible with CAM and Grad-CAM naturally.
    """
    def __init__(self, num_classes=2):
        super(VanillaCNN, self).__init__()
        # Input shape: [batch_size, 3, H, W]
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H, W -> H/2, W/2
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2 -> H/4, W/4
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4 -> H/8, W/8
            
            # Block 4 (Final Convolution layer for CAM)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Global Average Pooling to reduce spatial dimensions to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Linear classifier
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_resnet_model(pretrained=True, num_classes=2):
    """
    Loads ResNet-18 model, either pretrained on ImageNet or initialized randomly.
    Modifies the final fully connected layer for binary classification.
    """
    if pretrained:
        # Load with default pretrained weights
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
        
    # Replace the FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_inception_model(pretrained=True, num_classes=2):
    """
    Loads GoogLeNet (Inception v1) model from torchvision.
    We set aux_logits=False to avoid auxiliary output issues during training and inference.
    """
    if pretrained:
        weights = models.GoogLeNet_Weights.DEFAULT
        model = models.googlenet(weights=weights, aux_logits=True)
        model.aux_logits = False
    else:
        model = models.googlenet(weights=None, aux_logits=False)
        
    # Replace the FC classifier layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_mobilenet_model(pretrained=True, num_classes=2):
    """
    Loads MobileNet V2 model, which uses depthwise separable convolutions
    for efficiency. Modifies the classifier block.
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)
        
    # Replace the classification head (classifier[1] is the linear layer)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_model(model_name, pretrained=True, num_classes=2):
    """
    Factory function to retrieve the desired model.
    """
    model_name = model_name.lower().strip()
    if model_name == "vanilla":
        # Vanilla is always trained from scratch
        return VanillaCNN(num_classes=num_classes)
    elif model_name == "resnet":
        return get_resnet_model(pretrained=pretrained, num_classes=num_classes)
    elif model_name == "inception":
        return get_inception_model(pretrained=pretrained, num_classes=num_classes)
    elif model_name == "mobilenet":
        return get_mobilenet_model(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: vanilla, resnet, inception, mobilenet")


if __name__ == "__main__":
    # Test instantiating the models
    dummy_input = torch.randn(2, 3, 128, 128)
    
    for name in ["vanilla", "resnet", "inception", "mobilenet"]:
        model = get_model(name, pretrained=False)
        output = model(dummy_input)
        print(f"Model: {name:10s} | Output shape: {list(output.shape)}")
