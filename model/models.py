import torch.nn as nn
from torchvision import models
import timm

class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(InceptionV3Classifier, self).__init__()
        self.inception_v3 = models.inception_v3(weights='DEFAULT' if pretrained else None)
        in_features = self.inception_v3.fc.in_features
        self.inception_v3.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.inception_v3(x)
    
class InceptionResNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(InceptionResNetV2Classifier, self).__init__()
        self.inceptionresnet_v2 = models.inceptionresnetv2(weights='DEFAULT' if pretrained else None)
        in_features = self.inceptionresnet_v2.fc.in_features
        self.inceptionresnet_v2.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.inceptionresnet_v2(x)

class XceptionClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionClassifier, self).__init__()
        self.xception = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.xception(x)
    
class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DenseNetClassifier, self).__init__()
        self.densenet = timm.create_model('densenet121', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.densenet(x)