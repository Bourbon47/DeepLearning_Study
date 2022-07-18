#CNN with 'pytorch' library + https://inhovation97.tistory.com/37
#Ready for data set and path
import os

classes = ['buildings', 'forests', 'glacier', 'mountains', 'sea', 'street']
train_path = 'train/'
test_path = 'test/'

print("[ train dataset ]")
for i in range(6):
    print(f'class {i}의 개수: {len(os.listdir(train_path + classes[i]))}')

print("\n[ test dataset ]")
for i in range(6):
    print(f'class {i}의 개수: {len(os.listdir(test_path + classes[i]))}')
    
#Image Augmentation and Normalization
#Augmentation(증식) : 데이터 수가 많지 않을 때, 이미지의 색, 각도 등을 약간씩 변형하여 data의 수를 늘림
# ++) 학습 이미지의 수가 적어서 overfitting이 발생할 가능성을 줄이기 위해 기존 훈련 데이터로부터 그럴듯한 이미지를 랜덤하게 생성하여 데이터의 수를 늘림.
# ++) epoch수를 늘려주는데 부담을 줄인다
# ++) train_set에만 적용
# RandomResized() : 학습 시 Random으로 이미지의 일부를 잘라내어 특정 크기로 변환
# RandomHorizontalFlip() : 학습 시 Random으로 이미지를 좌우 반전
#Normalization(정규화) : 범위를 조정함으로써 step해 나가는 'landscape를 안정화 시켜서 local optima 문제를 예방'하고, 속도 측면에서도 좋아진다고합니다.
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available.")


transform_train = transforms.Compose([
    transforms.RandomResizedCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(train_path, transform_train)
test_dataset = datasets.ImageFolder(test_path, transform_test)

train_data_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=2
)

test_data_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=2
)

print('Training dataset size:', len(train_dataset))
print('Validation dataset size:', len(test_dataset))

class_names = train_dataset.classes
print('Class names:', class_names)

#Image Visualization
import numpy as np
import matplotlib.pyplot as plt
import torchvision

plt.rcParams['figure.figsize'] = [15, 9]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({
    'font.size': 20
})

def imshow(image, title):
    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.title(title)
    plt.show()


# get one batch
iterator = iter(train_data_loader)

# image visualization
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])
imshow(out, title=[class_names[x] for x in classes[:4]])









#Inline Practice 1
#LeNet 구조 정의Permalink
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding)
#F.max_pool2d(input, kernel_size, stride)
'''
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(8450, 500) #50*50*13
        self.fc2 = nn.Linear(500, 6)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = x.view(-1, 8450)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

import torchsummary

_device = device.type

model = LeNet().to(device)
torchsummary.summary(model, (3, 64, 64), device=_device)
'''

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 20, kernel_size = 5, stride = 1)
        self.maxp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 5, stride = 1)
        self.maxp2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(in_features = 50*13*13, out_features = 500)
        self.Re1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features = 500, out_features = 6)
                                      
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        
        x = self.maxp1(x)
        
        x = self.conv2(x)
        x = torch.tanh(x)
        
        x = self.maxp2(x)
        
        x = x.view(-1, 50*13*13)
        
        x = self.fc1(x)
        x = torch.tanh(x)
        
        x = self.Re1(x)
        
        x = self.fc2(x)
        x = torch.tanh(x)
        
        return x
    
import torchsummary

_device = device.type

model = LeNet().to(device)
torchsummary.summary(model, (3, 64, 64), device=_device)




















#Ready for Train and Test Function 
def train(model, epoch, optimizer, criterion, data_loader):
    print(f"[train epoch : {epoch}]")
    model.train()

    losses = 0.
    correct = 0
    total = 0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f"train accuracy : {100. * (correct / total):.3f}")
    print(f"train average loss : {losses / total:.3f}")

    return (100. * correct / total, losses / total)

def test(model, epoch, criterion, data_loader):
    print(f"[test epoch : {epoch}]")
    model.eval()

    losses = 0.
    correct = 0
    total = 0    
    
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()        

    print(f"test accuracy : {100. * (correct / total):.3f}")
    print(f"test average loss : {losses / total:.3f}")

    return (100. * correct / total, losses / total)




























#Inline Practice 2
#epoch : 30, learning_rate = 0.0005 (5e-4), criterion(loss function) : crossEntropy, optimizer : Adam
import time
import torch.optim as optim

model = LeNet()
model = model.to(device)


'''
input options
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
epochs = 30

train_result = []
test_result = []

start_time = time.time()
for epoch in range(epochs):
    '''
    train, test
    '''
    train_acc , train_loss = train(model, epoch, optimizer, criterion, train_data_loader)
    test_acc , test_loss = test(model, epoch, criterion, test_data_loader)


    # model_save
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(model.state_dict(), './checkpoint/model.pt')
    print(f"model save. ({time.time() - start_time:.3f}sec)\n\n")

    train_result.append((train_acc, train_loss))
    test_result.append((test_acc, test_loss))

























#Inline Practice 3
import matplotlib.pyplot as plt

num_epochs = epochs

plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()































#Inline Practice 4
import torch.nn as nn
import torch.nn.functional as F

class CustomLeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=30, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=60*13*13, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=6),
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

import torchsummary

_device = device.type

model = CustomLeNet().to(device)
torchsummary.summary(model, (3, 64, 64), device=_device)



