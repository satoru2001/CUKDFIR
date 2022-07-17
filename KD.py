from utils.tools import *
from network import *
import numpy as np
import torch, torchvision
from torchvision import datasets, transforms
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import time
from datetime import datetime
import pytz
IST = pytz.timezone('Asia/Kolkata')

#Macros
student_ver = 'v1'
# student = 'v2'
# data_set = 'nus_wide_m'
data_set = 'cifar10'


#Networks
class AlexNet(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)


    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        return x

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out

class Student_resnet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
  
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, layers[4], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7
        x = self.layer5(x)          # 3*3
        x = x.view(x.size(0), -1)     # convert 1 X 1 to vector
        return x

def student_resnet_model():
    layers=[2, 3, 5, 3, 2]
    model = Student_resnet(BasicBlock, layers)
    return model

class Student_alexnet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
  
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 256, layers[4], stride=2)
        self.binary_new = nn.Linear(4096,hash_size)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7
        x = self.layer5(x)          # 3*3
        x = x.view(x.size(0), -1)     # convert 1 X 1 to vector

        return x

def student_alexnet_model():
    layers=[2, 3, 5, 3, 2]
    model = Student_alexnet(BasicBlock, layers)
    return model

#Data Loaders

#cifar 10
def image_transform(resize_size=256, crop_size=224, data_set="train_set"):
    step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])
#Nus wide
if data_set == "nus_wide_m":
    dataset = torchvision.datasets.ImageFolder('./dataset/nus_wide_m', transform=image_transform())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    epochs = 120
    lr=0.000003
elif data_set == "cifar10":
    dataset = torchvision.datasets.CIFAR10(root='./dataset/cifar', download=True, transform=image_transform())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                          shuffle=True,pin_memory=True)
    epochs = 160
    lr=0.0001


#Training
device = "cuda:0"
if student_ver == 'v1':
    teacher = ResNet().to(device)
    student = student_resnet_model().to(device)
elif student_ver == 'v2':
    teacher = AlexNet().to(device)
    student = student_alexnet_model().to(device)

#Making teacher non trainable
for param in teacher.parameters():
    param.requires_grad = False

criterion = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(student.parameters(),lr)
                                                                                               
# sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, min_lr=1e-5, verbose=False)
datetime_ist = datetime.now(IST)
current_time = datetime_ist.strftime('%Y:%m:%d %H:%M:%S %Z %z')                                                                                                                                                     
print("Training Started at Time %s" % (current_time))
student.train()
teacher.eval()

for i in range(epochs):
  train_loss = 0
  with tqdm(dataloader, unit="batch") as tepoch:
    for [image,_] in tepoch:
        tepoch.set_description(f"Epoch {i}")
        image = image.to(device)
        teacher_output = teacher(image)
        student_output = student(image)
        loss = criterion(student_output,teacher_output)
        optimizer.zero_grad()
        tepoch.set_postfix(loss=loss.item())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    datetime_ist = datetime.now(IST)
    current_time = datetime_ist.strftime('%Y:%m:%d %H:%M:%S %Z %z')
    train_loss = train_loss / len(dataloader)
    print("loss:%.3f Epoch %d" % (train_loss, i))
torch.save(student.state_dict(), f"./KD_Checkpoints/student_{student_ver}.pth")