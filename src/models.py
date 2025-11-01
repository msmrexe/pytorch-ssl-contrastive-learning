import torch
import torch.nn as nn


class Network(nn.Module):
    """
    A simple convolutional neural network for baseline image classification.
    Architecture:
    - Backbone: 3 Conv/BN/ReLU/MaxPool blocks
    - FC: 1 Linear/ReLU/Dropout block
    - Head: 1 Linear classification layer
    """
    def __init__(self, num_classes=10):
        super(Network, self).__init__()
        self.backbone = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 x 16 x 16

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 8 x 8

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x 4 x 4
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512), # 2048 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.classification_head = nn.Linear(512, num_classes)

    def extract_features(self, x):
        """Extracts features from the backbone."""
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        x = self.classification_head(x)
        return x


class Supervised_Learning(nn.Module):
    """
    A multi-head model for the self-supervised pretext task.
    Uses the *same* backbone as the baseline model for a fair comparison.
    - Backbone: Identical to baseline
    - FC: Shared Linear/ReLU/Dropout layer
    - Heads: Four separate linear heads for:
        1. Rotation (4 classes)
        2. Shear (3 classes)
        3. Color (3 classes)
        4. Classification (10 classes)
    """
    def __init__(self, num_classes=10):
        super(Supervised_Learning, self).__init__()
        # Use the EXACT same backbone as the baseline model
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4x4
        )

        # Shared fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Heads for each pretext task and the final classification
        self.rotation_head = nn.Linear(512, 4)
        self.shear_head = nn.Linear(512, 3)
        self.color_head = nn.Linear(512, 3)
        self.classification_head = nn.Linear(512, num_classes)

    def extract_features(self, x):
        """Extracts features from the backbone."""
        return self.backbone(x)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1) # Flatten
        shared_features = self.fc(features)

        # Get predictions from each head
        rotation_pred = self.rotation_head(shared_features)
        shear_pred = self.shear_head(shared_features)
        color_pred = self.color_head(shared_features)
        class_pred = self.classification_head(shared_features)

        return rotation_pred, shear_pred, color_pred, class_pred


class SimSiam(nn.Module):
    """
    Implements the SimSiam model for contrastive learning.
    Architecture:
    - Encoder (f):
        - Backbone (same as baseline)
        - Projector (3-layer MLP)
    - Predictor (h): 2-layer MLP
    
    Forward pass computes z = f(x) and p = h(z).
    """
    def __init__(self, backbone_output_dim=2048, proj_dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()
        
        # 1. Encoder Backbone (same as previous parts)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 2. Encoder Projection MLP (f)
        # 128 * 4 * 4 = 2048
        feature_dim = 128 * 4 * 4
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, backbone_output_dim), nn.BatchNorm1d(backbone_output_dim), nn.ReLU(inplace=True),
            nn.Linear(backbone_output_dim, backbone_output_dim), nn.BatchNorm1d(backbone_output_dim), nn.ReLU(inplace=True),
            nn.Linear(backbone_output_dim, proj_dim), nn.BatchNorm1d(proj_dim)
        )

        # 3. Predictor MLP (h)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim), nn.BatchNorm1d(pred_dim), nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim)
        )

    def extract_features(self, x):
        """Extracts features from the backbone ONLY."""
        return self.backbone(x)

    def forward(self, x):
        # Pass through backbone and flatten
        features = self.backbone(x).view(x.size(0), -1)
        # Get projection z
        z = self.projector(features)
        # Get prediction p
        p = self.predictor(z)
        return z, p


def negative_cosine_similarity_stopgradient(p, z):
    """
    SimSiam loss function. Calculates negative cosine similarity.
    Applies stop-gradient to z.
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    z = z.detach() # Stop gradient
    return -cos(p, z).mean()


class ClassificationModel(nn.Module):
    """
    A linear classifier to be placed on top of the frozen SimSiam backbone.
    This is used for "linear probing".
    """
    def __init__(self, simsiam_model, num_classes=10):
        super(ClassificationModel, self).__init__()
        # Load the pre-trained backbone
        self.backbone = simsiam_model.backbone
        
        # Freeze the backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Add a new classification head (matches baseline's FC + Head)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # The backbone is frozen, no gradients are computed here
        with torch.no_grad():
            features = self.backbone(x).view(x.size(0), -1)
        # Only the classifier is trained
        return self.classifier(features)
