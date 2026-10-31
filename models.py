"""
models.py - Architectures for BloodMNIST Benchmark
Group 13: Aldrich Taqi Marvel, Darren Christian Rapang, Johannes Christian Tahun
Course: Deep Learning Final Project
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ==============================================================================
# 1. BASELINE MODEL: Custom 3-Stage CNN (Trained from Scratch)
# ==============================================================================
class BaselineCNN(nn.Module):
    """
    A custom convolutional network designed specifically for 28x28 input resolution.
    Features: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d progression,
    followed by Global Average Pooling and a Multi-Layer Classifier with Dropout.
    """
    def __init__(self, num_classes=8):
        super(BaselineCNN, self).__init__()
        
        # Block 1: Input (3, 28, 28) -> Output (32, 14, 14)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Input (32, 14, 14) -> Output (64, 7, 7)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Input (64, 7, 7) -> Output (128, 7, 7)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification Head: Input (128) -> Output (num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

    def get_layer_shapes(self, input_shape=(1, 3, 28, 28)):
        """Returns layer-by-layer shape progression for the IEEE report."""
        shapes = []
        x = torch.zeros(input_shape)
        shapes.append(("Input", list(x.shape)))
        
        x = self.block1(x)
        shapes.append(("Block 1 (Conv-BN-ReLU-Pool)", list(x.shape)))
        
        x = self.block2(x)
        shapes.append(("Block 2 (Conv-BN-ReLU-Pool)", list(x.shape)))
        
        x = self.block3(x)
        shapes.append(("Block 3 (Conv-BN-ReLU-AvgPool)", list(x.shape)))
        
        x = self.classifier[0](x)
        shapes.append(("Flatten", list(x.shape)))
        
        x = self.classifier[1:4](x)
        shapes.append(("Dense (64 units) + Dropout(0.3)", list(x.shape)))
        
        x = self.classifier[4](x)
        shapes.append(("Output Layer (8 classes)", list(x.shape)))
        
        return shapes


# ==============================================================================
# 2. METHOD A: Standard Pretrained ResNet-18 (Transfer Learning)
# ==============================================================================
def get_method_a_resnet18(num_classes=8):
    """
    Standard transfer learning using torchvision's ResNet-18 (ImageNet weights).
    Modifies the initial convolution to 3x3 with stride 1 and removes initial
    maxpool to preserve spatial fidelity on 28x28 medical images.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Adapt first layer for 28x28 resolution
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool to prevent over-downsampling
    
    # Replace final linear layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


# ==============================================================================
# 3. METHOD B: Squeeze-and-Excitation Enhanced ResNet-18 (SE-ResNet18)
# ==============================================================================
class SqueezeExcitation(nn.Module):
    """
    Channel-Attention Block (Hu et al., CVPR 2018).
    Computes adaptive channel weights to emphasize diagnostic cytological features.
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class SEBasicBlock(nn.Module):
    """
    Residual block with integrated Squeeze-and-Excitation attention.
    Formula: y = x + SE(F(x))
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels, reduction=16)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply Channel Attention
        out += residual     # Residual Skip Connection
        out = self.relu(out)
        return out


def get_method_b_se_resnet18(num_classes=8):
    """
    Method B: ResNet-18 with Squeeze-and-Excitation attention attached to all residual stages.
    """
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Adapt first layer
    base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    base_model.maxpool = nn.Identity()

    # Wrap each layer with SE attention
    for layer in [base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4]:
        for block in layer:
            channels = block.conv2.out_channels
            block.add_module("se", SqueezeExcitation(channels, reduction=16))
            
            # Monkey-patch block forward to include SE
            original_forward = block.forward
            def make_new_forward(b):
                def forward(x):
                    identity = x
                    out = b.conv1(x)
                    out = b.bn1(out)
                    out = b.relu(out)
                    out = b.conv2(out)
                    out = b.bn2(out)
                    out = b.se(out)
                    if b.downsample is not None:
                        identity = b.downsample(x)
                    out += identity
                    out = b.relu(out)
                    return out
                return forward
            block.forward = make_new_forward(block)

    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, num_classes)
    
    return base_model

def count_parameters(model):
    """Calculates total trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    base = BaselineCNN()
    res = get_method_a_resnet18()
    se_res = get_method_b_se_resnet18()
    
    print(f"Baseline Params: {count_parameters(base):,}")
    print(f"Method A Params: {count_parameters(res):,}")
    print(f"Method B Params: {count_parameters(se_res):,}")
    
    print("\n[Baseline Layer Shapes]")
    for name, shape in base.get_layer_shapes():
        print(f"  {name:35s}: {shape}")
