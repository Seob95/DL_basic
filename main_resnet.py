import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import torch.backends.cudnn as cudnn

import torch.optim as optim
from torchsummary import summary 
from tensorboardX import SummaryWriter 
writer = SummaryWriter('runs/graph') 

os.environ["CUDA_VISIBLE_DEVICES"] = '0'                # GPU Number 
start_time = time.time()
batch_size = 128                                         
learning_rate = 0.005
root_dir = 'drive/app/cifar10/'
default_directory = './runs/save_models'

from dropblock import DropBlock2D, LinearScheduler                           ###!!! dropblock
from RandAugment import RandAugment
from CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
drop_prob = 0.8                                                             #!! drop block
block_size = 5                                                              #!! drop block


# Data Augmentation
transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32, padding=4),               # Random Position Crop
    RandAugment(5,20),                                   ###!!! RandAugment 추가 (3,5): 94.69 (3,10): 94.97
    transforms.RandomHorizontalFlip(),                  # right and left flip
    transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
])

transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
])

# transform_train = transforms.Compose([                                                                           ##!! Image resize 실험
#     transforms.Resize((64,64)),
#     transforms.RandomCrop((64,64)),               # Random Position Crop
#     #RandAugment(3,5),
#     transforms.RandomHorizontalFlip(),                  # right and left flip
#     transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
#     transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
#                          std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
# ])

# transform_test = transforms.Compose([
#     transforms.Resize((64,64)),
#     transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
#     transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
#                          std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
# ])



# automatically download
train_dataset = datasets.CIFAR10(root=root_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=True)

test_dataset = datasets.CIFAR10(root=root_dir,
                                train=False,
                                transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,pin_memory=True,            # at Training Procedure, Data Shuffle = True
                                           num_workers=0)           # CPU loader number

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,pin_memory=True,            # at Test Procedure, Data Shuffle = False
                                          num_workers=0)            # CPU loader number
print('@@@@@@@@ train dataset :', train_dataset)
print('@@@@@@@@ Batch_size :', batch_size)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))                                   #! relu -> gelu로 변경
        out = out + self.shortcut(x)
        out = F.elu(out)                                               #! relu -> gelu로 변경
        return out
    
class BottleNeck(nn.Module): 
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),                                              #! relu -> GELU로 변경
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.elu = nn.ELU()                               #! relu -> ELU로 변경

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.ELU(x)                                                             #! relu -> ELU로 변경
        return x  #!#

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64                                                                                 #! 64 -> 128

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)                   #! 64 -> 128
        self.bn1 = nn.BatchNorm2d(64)                                                                   #! 64 -> 128
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)                              #! 64 -> 128
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)                             #! 128 -> 256
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)                             #! 256 -> 512
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)                            #! 512 -> 1024
        self.linear = nn.Linear(512*block.expansion, num_classes)                            #! 512 -> 1024

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))                                   #! relu -> elu로 변경
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



#####!!!!!!! resNext start
class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))           
        out = F.relu(self.bn2(self.conv2(out)))         
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)                              
        return out


class ResNeXt(nn.Module):
    #def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10, drop_prob = 0.5, block_size = 5):   #! drop block
        super(ResNeXt, self).__init__()
        self.dropblock = LinearScheduler(DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0., stop_value=drop_prob, nr_steps=5e3)                                                     #! drop block
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        self.dropblock.step()  # increment number of iterations             #! drop block

        out = F.relu(self.bn1(self.conv1(x)))      
        out = self.dropblock(self.layer1(out))                           #! drop block
        # out = self.dropblock(self.layer2(out))                           #! drop block
        #out = self.layer1(out)
        out = self.layer2(out)                                             
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)                  #! pooling 바꿔서 64x64 실험해보기             8->16으로 변경
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
#########!!!!!!!!!!!! resNext end

#model = ResNet(BasicBlock, [2, 2, 2, 2])               #! resnet 18
#model = ResNet(BottleNeck, [3, 4, 6, 3])            #! resnet 50

#model = ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64)             #! ResNext 2x64d
model = ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, drop_prob=drop_prob, block_size=block_size)             #! ResNext 2x64d drop block


optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)                        #! weight_decay=1e-4 -> ResNext 할때 5e-4로 변경

scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10000, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)                     ####! lr scheduler 첨엔 T_0 = 100이였음

criterion = nn.CrossEntropyLoss()                                                                                               ###!!! 기존 criterion

if torch.cuda.device_count() > 0:
    print("USE", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
else:
    print("USE ONLY CPU!")


def train(epoch):
    model.train()
    train_loss = 0 
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()                                                            ####! lr scheduler 추가함


        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)

        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            
            writer.add_scalar('training_loss', (train_loss / (batch_idx + 1)) , epoch * len(train_loader) + batch_idx)              ####!
            writer.add_scalar('training_accuracy', (100. * correct / total), epoch * len(train_loader) + batch_idx)                 ####!
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)                         ####!

def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        outputs = model(data)
        loss = criterion(outputs, target)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        
        writer.add_scalar('test_loss', test_loss / (batch_idx + 1), epoch * len(test_loader)+ batch_idx)                         ####!
        writer.add_scalar('test_accuracy', 100. * correct / total, epoch * len(test_loader)+ batch_idx)                         ####!
        
    print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def save_checkpoint(directory, state, filename='latest.tar.gz'):

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_filename = os.path.join(directory, filename)
    torch.save(state, model_filename)
    print("=> saving checkpoint")

def load_checkpoint(directory, filename='latest.tar.gz'):

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        return None

start_epoch = 0

checkpoint = load_checkpoint(default_directory)
if not checkpoint:
    pass
else:
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for epoch in range(start_epoch, 165):                       #!                     

    if epoch < 80:
        lr = learning_rate
    elif epoch < 120:
        lr = learning_rate * 0.1
    else:
        lr = learning_rate * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train(epoch)
    save_checkpoint(default_directory, {
        'epoch': epoch,
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    })
    test()

start_epoch = 0

# for epoch in range(start_epoch, 1):    
#     test() 

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))