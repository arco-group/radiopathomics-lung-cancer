import torchvision.models as models
import torch.nn as nn

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
class vgg_custom(nn.Module):
    def __init__(self, originalModel, num_classes, path_dim, mode):
        super(vgg_custom, self).__init__()
        self.mode = mode
        
        self.features = originalModel.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )
        self.linear = nn.Linear(path_dim, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        out = self.linear(features)
        
        if self.mode == "classification":
            return out
        elif self.mode == "feature_extraction":
            return features
        
class mobilenet_v2_custom(nn.Module):
    def __init__(self, originalModel, num_classes, path_dim, mode):
        super(mobilenet_v2_custom, self).__init__()
        self.mode = mode
        
        self.features = originalModel.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=path_dim, bias=True)
            
        )
        self.linear = nn.Linear(path_dim, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        out = self.linear(features)
        
        if self.mode == "classification":
            return out
        elif self.mode == "feature_extraction":
            return features

class resnet18_custom(nn.Module):
    def __init__(self, originalModel, num_classes, path_dim, mode):
        super(resnet18_custom, self).__init__()
        self.mode = mode
        
        self.conv1 = originalModel.conv1
        self.bn1 = originalModel.bn1
        self.relu = originalModel.relu
        self.maxpool = originalModel.maxpool
        self.layer1 = originalModel.layer1
        self.layer2 = originalModel.layer2
        self.layer3 = originalModel.layer3
        self.layer4 = originalModel.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(in_features=512, out_features=path_dim, bias=True)
        self.fc2 = nn.Linear(in_features=path_dim, out_features=num_classes, bias=True)        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)     
        x = x.view(x.size(0), -1)
        
        features = self.fc1(x)
        out = self.fc2(features)
        
        if self.mode == "classification":
            return out
        elif self.mode == "feature_extraction":
            return features

class alexnet_custom(nn.Module):
    def __init__(self, originalModel, num_classes, path_dim, mode):
        super(alexnet_custom, self).__init__()
        
        self.mode = mode
        
        self.features = originalModel.features
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256*6*6, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=path_dim, bias=True)
        )
        self.linear = nn.Linear(path_dim, num_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        out = self.linear(features)
        
        if self.mode == "classification":
            return out
        elif self.mode == "feature_extraction":
            return features    
        
class googlenet_custom(nn.Module):
    def __init__(self, originalModel, num_classes, path_dim, mode):
        super(googlenet_custom, self).__init__()
        
        self.mode = mode
        
        self.conv1 = originalModel.conv1
        self.maxpool1 = originalModel.maxpool1
        self.conv2 = originalModel.conv2
        self.conv3 = originalModel.conv3
        self.maxpool2 = originalModel.maxpool2
        self.inception3a = originalModel.inception3a
        self.inception3b = originalModel.inception3b
        self.maxpool3 = originalModel.maxpool3
        self.inception4a = originalModel.inception4a
        self.inception4b = originalModel.inception4b
        self.inception4c = originalModel.inception4c
        self.inception4d = originalModel.inception4d
        self.inception4e = originalModel.inception4e
        self.maxpool4 = originalModel.maxpool4
        self.inception5a = originalModel.inception5a
        self.inception5b = originalModel.inception5b
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        
        self.fc1 = nn.Linear(in_features=1024, out_features=path_dim, bias=True)
        self.fc2 = nn.Linear(path_dim, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        features = self.fc1(x)
        out = self.fc2(features)
        
        if self.mode == "classification":
            return out
        elif self.mode == "feature_extraction":
            return features      


def get_model(net_name, num_classes=2, path_dim=32, mode = "classification"):
    
    if net_name == "vgg19":
        vgg19 = models.vgg19(pretrained=True)
        set_parameter_requires_grad(vgg19, True)

        model_ft = vgg_custom(vgg19, num_classes = num_classes, path_dim = path_dim, mode = mode)
        
    elif net_name == "googlenet":
        googlenet = models.googlenet(pretrained=True)
        set_parameter_requires_grad(googlenet, True)
        
        model_ft = googlenet_custom(googlenet, num_classes = num_classes, path_dim = path_dim, mode = mode)
        
    elif net_name == "alexnet":
        alexnet = models.alexnet(pretrained=True)
        set_parameter_requires_grad(alexnet, True)
        
        model_ft = alexnet_custom(alexnet, num_classes = num_classes, path_dim = path_dim, mode = mode)
    elif net_name == "resnet18":
        resnet18 = models.resnet18(pretrained=True)
        set_parameter_requires_grad(resnet18, True)
        
        model_ft = resnet18_custom(resnet18, num_classes = num_classes, path_dim = path_dim, mode = mode)
    elif net_name == "mobilenet_v2":
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        set_parameter_requires_grad(mobilenet_v2, True)
        
        model_ft = mobilenet_v2_custom(mobilenet_v2, num_classes = num_classes, path_dim = path_dim, mode = mode)
    
    return model_ft