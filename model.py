import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    """
    Residual Block implementation for deep neural networks.
    
    This block implements the skip connection architecture that allows gradients
    to flow directly through the network, enabling training of very deep networks.
    The block consists of two convolutional layers with batch normalization and
    a shortcut connection that adds the input to the output.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the Residual Block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for convolution operations (default: 1)
        """
        super().__init__()
        
        # First convolutional layer with 3x3 kernel
        # Uses stride and padding=1 to maintain spatial dimensions (when stride=1)
        # bias=False because BatchNorm will handle the bias term
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        
        # First batch normalization layer
        # Normalizes the output of conv1 to stabilize training
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer with 3x3 kernel
        # The stride is only applied to the first conv layer in a residual block
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        # Second batch normalization layer
        # Normalizes the output of conv2 before adding the shortcut
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Initialize shortcut connection as identity (empty sequential)
        self.shortcut = nn.Sequential()
        
        # Determine if we need a learnable shortcut connection
        # This is needed when dimensions change (different stride or channel count)
        self.use_shortcut = stride != 1 or in_channels != out_channels
        
        if self.use_shortcut:
            # Create a 1x1 convolution to match dimensions between input and output
            # This allows the skip connection to work when channel counts differ
            # or when spatial dimensions are reduced by stride > 1
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, fmap_dict=None, prefix=""):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor after residual connection and activation
        """
        # First convolutional path
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        # Second convolutional path
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Compute shortcut connection
        # If dimensions match, use identity; otherwise use learned projection
        shortcut = self.shortcut(x) if self.use_shortcut else x

        # Add shortcut to main path (the key residual connection)
        out_add = out + shortcut

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add
        
        # Apply final ReLU activation after the addition
        out = torch.relu(out_add)

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out

        return out
    

class AudioCNN(nn.Module):
    """
    Audio Classification CNN based on ResNet architecture.
    
    This network is designed for audio classification tasks, specifically
    for processing mel-spectrograms. It uses residual blocks to enable
    deep feature learning while maintaining gradient flow.
    """
    
    def __init__(self, num_classes=50):
        """
        Initialize the Audio CNN model.
        
        Args:
            num_classes (int): Number of output classes for classification (default: 50 for ESC-50)
        """
        super().__init__()
        
        # Initial convolutional stem
        # Large 7x7 kernel with stride=2 for initial feature extraction and downsampling
        # Followed by batch norm, ReLU activation, and max pooling for further downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),  # Input: 1 channel (mono audio)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(3, stride=2, padding=1)  # Further spatial reduction
        )
        
        # Residual layers with increasing channel counts
        # Each layer consists of multiple residual blocks
        
        # Layer 1: 3 blocks, 64 channels, no downsampling
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)])
        
        # Layer 2: 4 blocks, 128 channels, first block downsamples with stride=2
        self.layer2 = nn.ModuleList([
            ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)
        ])
        
        # Layer 3: 6 blocks, 256 channels, first block downsamples with stride=2
        self.layer3 = nn.ModuleList([
            ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)
        ])
        
        # Layer 4: 3 blocks, 512 channels, first block downsamples with stride=2
        self.layer4 = nn.ModuleList([
            ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)
        ])
        
        # Global Average Pooling
        # Reduces spatial dimensions to 1x1, creating a feature vector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        # Dropout for regularization
        # Randomly sets 50% of features to zero during training to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
        # Final classification layer
        # Maps 512-dimensional feature vector to class probabilities
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_maps=False):
        """
        Forward pass through the audio CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
                             Typically a mel-spectrogram with 1 channel
            
        Returns:
            torch.Tensor: Logits for each class (batch_size, num_classes)
        """
        if not return_feature_maps:
            # Initial convolution and pooling
            x = self.conv1(x)
            
            # Pass through each residual layer
            # Each layer processes features at different scales and complexities
            for block in self.layer1:
                x = block(x)
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)
            
            # Global average pooling to create fixed-size feature vector
            # Converts (batch_size, 512, H, W) to (batch_size, 512, 1, 1)
            x = self.avgpool(x)
            
            # Flatten to 1D feature vector
            # Converts (batch_size, 512, 1, 1) to (batch_size, 512)
            x = x.view(x.size(0), -1)
            
            # Apply dropout for regularization (only during training)
            x = self.dropout(x)
            
            # Final linear classification layer
            # Produces logits for each class
            x = self.fc(x)

            return x
        else:
            feature_maps = {}

            # Initial convolution and pooling
            x = self.conv1(x)

            feature_maps["conv1"] = x
            
            # Pass through each residual layer
            # Each layer processes features at different scales and complexities
            for i, block in enumerate(self.layer1):
                x = block(x, feature_maps, prefix=f"layer1.block{i}")
            feature_maps["layer1"] = x

            for i, block in enumerate(self.layer2):
                x = block(x, feature_maps, prefix=f"layer2.block{i}")
            feature_maps["layer2"] = x

            for i, block in enumerate(self.layer3):
                x = block(x, feature_maps, prefix=f"layer3.block{i}")
            feature_maps["layer3"] = x

            for i, block in enumerate(self.layer4):
                x = block(x, feature_maps, prefix=f"layer4.block{i}")
            feature_maps["layer4"] = x
            
            # Global average pooling to create fixed-size feature vector
            # Converts (batch_size, 512, H, W) to (batch_size, 512, 1, 1)
            x = self.avgpool(x)
            
            # Flatten to 1D feature vector
            # Converts (batch_size, 512, 1, 1) to (batch_size, 512)
            x = x.view(x.size(0), -1)
            
            # Apply dropout for regularization (only during training)
            x = self.dropout(x)
            
            # Final linear classification layer
            # Produces logits for each class
            x = self.fc(x)

            return x, feature_maps