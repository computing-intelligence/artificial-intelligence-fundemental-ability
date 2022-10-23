import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch import nn
from torch import optim


preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

cifar_10 = torchvision.datasets.CIFAR10('.', download=True, transform=preprocessing)
train_loader = torch.utils.data.DataLoader(cifar_10,
                                          batch_size=12,
                                          shuffle=True)

res_net = models.resnet18(pretrained=False)


plt.imshow(cifar_10[10][0].permute(1, 2, 0))

for param in res_net.parameters():
    param.requires_grad = False  # Use Pre-trained Paramters Directly

# ResNet: CNN(with residual)-> CNN(with residual)-CNN(with residual)-Fully Connected

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = res_net.fc.in_features

res_net.fc = nn.Linear(num_ftrs, 10) # only update this part parameters 
# 整个的resnet的权重，除了最后一行是需要进行梯度更新的，别的都不更新
# 之前如果我们不适用已经训练好的模型，那么我们需要拟合的参数是整个模型里的参数
# 非常非常多, 在百万级别
# 我们只需要你和最后一层只需要拟合5000多个参数

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(res_net.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochslosses = []

epochs = 10


for epoch in range(epochs):
    loss_train = 0
    for i, (imgs, labels) in enumerate(train_loader):        
        print(i)
        outputs = res_net(imgs)
        
        loss = criterion(outputs, labels)
        
        optimizer_conv.zero_grad()
        
        loss.backward() # -> only update fully connected layer
        
        optimizer_conv.step()
        
        loss_train += loss.item()
        
        #if i > 0 and i % 10 == 0:
        print('Epoch: {}, batch: {}'.format(epoch, i+1))
        print('-- loss: {}'.format(loss_train / (i+1)))
            
    losses.append(loss_train / len(train_loader))


