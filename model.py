import torch, torchvision, torch.nn as nn,torch.nn.functional as F,time,os
from torchsummary import summary


class Unet(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        
        self.conv1c = nn.Conv2d(1,64,kernel_size = 3,stride =1 ,padding = 1)
        self.relu1 = nn.ReLU(inplace = True)
        self.drop = nn.Dropout2d(.5)
        self.mp1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        
        self.conv2c = nn.Conv2d(64,128,kernel_size = 3,stride =1 ,padding = 1)
        self.relu2 = nn.ReLU(inplace = True)
        self.mp2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
          
        self.conv3c = nn.Conv2d(128,128,kernel_size=3 ,stride=1,padding=1)
        self.norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.mup2 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.conv2d = nn.Conv2d(128,64,kernel_size = 3,stride =1 ,padding = 1)
        
        self.mup1 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.conv1d = nn.Conv2d(64,28,kernel_size = 3,stride =1 ,padding = 1)
        self.norm2 = nn.BatchNorm2d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1e = nn.Conv2d(28,1,kernel_size = 3,stride =1 ,padding = 1)
        self.soft = nn.Softmax2d()
    def forward(self,x):
        #x = self.relu1(self.conv1To3(x))
        x = self.relu1(self.conv1c(x))
        x = self.drop(x)
        x,idx1 = self.mp1(x)
        
        x = self.relu2(self.conv2c(x))
        x = self.drop(x)
        x,idx2 = self.mp2(x)
        
        x = self.conv3c(x)
        x = self.norm(x)
        
        x = self.mup2(x,idx2)
        x = self.drop(x)
        x = self.conv2d(x)
        
        x = self.mup1(x,idx1)
        x = self.drop(x) 
        x = self.conv1d(x)
        x = self.norm2(x)
        x = self.conv1e(x)
        x = self.soft(x)
        return x
class SegNet(nn.Module):
    def __init__(self):
        super(SegNet,self).__init__()
        #encode
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(9, 9), stride=(1, 1),padding=(4,4), bias=False)
        self.bnorm1 =nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1,1), bias=False)
        self.bnorm11 =nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1,1),bias=False)
        self.bnorm2 =nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2,  dilation=1, ceil_mode=False)
        
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1,1), bias=False)
        self.bnorm22 =nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(128,256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1),bias=False)
        self.bnorm3 =nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2,  dilation=1, ceil_mode=False)
        
        self.conv33 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1,1), bias=False)
        self.bnorm33 =nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1,1),bias=False)
        self.bnorm4 =nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2,  dilation=1, ceil_mode=False)
        
        self.conv44 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1,1), bias=False)
        self.bnorm44 =nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = nn.ReLU(inplace=True)
        #Decoder
        #self.mup1 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        #self.conv1d = nn.Conv2d(64,1,kernel_size=3,stride = 1,padding =1 )
        
        self.tconv1 = nn.ConvTranspose2d(512,256,kernel_size=(4, 4), stride=(2, 2), padding=(1,1),bias=False)
        self.bconv1 = nn.Conv2d(512,256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1),bias=False)
        self.brelu1 = nn.ReLU(inplace=True)
        
        self.tconv2 = nn.ConvTranspose2d(256,128,kernel_size=(4, 4), stride=(2, 2), padding=(1,1),bias=False)
        self.bconv2 = nn.Conv2d(256,128, kernel_size=(3, 3), stride=(1, 1), padding=(1,1),bias=False)
        self.brelu2 = nn.ReLU(inplace=True)
        
        self.tconv3 = nn.ConvTranspose2d(128,64,kernel_size=(4, 4), stride=(2, 2), padding=(1,1),bias=False)
        self.bconv3 = nn.Conv2d(128,64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1),bias=False)
        self.brelu3 = nn.ReLU(inplace=True)
        
        self.tconv4 = nn.ConvTranspose2d(64,1,kernel_size=(4, 4), stride=(2, 2), padding=(1,1),bias=False)
        self.bconv4 = nn.Conv2d(64,1, kernel_size=(3, 3), stride=(1, 1), padding=(1,1),bias=False)
        self.brelu4 = nn.Sigmoid()
        
    def forward(self,x):
        x = (self.bnorm1(self.relu1(self.conv1(x))))
        
        x = self.mp1(x)
        c2 = self.bnorm11(self.relu11(self.conv11(x)))
        
        x = (self.bnorm2(self.relu2(self.conv2(c2))))
        x = self.mp2(x)
        c3 = self.bnorm22(self.relu22(self.conv22(x)))      #128,128,128
        
        x = (self.bnorm3(self.relu3(self.conv3(c3))))
        x = self.mp3(x)
        c4 = self.bnorm33(self.relu33(self.conv33(x)))      #256, 64, 64
        
        x = (self.bnorm4(self.relu4(self.conv4(c4))))
        x = self.mp4(x)
        x = self.bnorm44(self.relu44(self.conv44(x)))      #512,32,32
        
        x = self.tconv1(x)
        x = torch.cat((x,c4),1)                       #512,64,64
        x = self.bconv1(x)
        x = self.brelu1(x)                              #256,64,64
        
        x = self.tconv2(x)
        x = torch.cat((x,c3),1)   
        x = self.bconv2(x)
        x = self.brelu2(x)                               #128,128,128
        
        x = self.tconv3(x)
        x = torch.cat((x,c2),1)   
        x = self.bconv3(x)
        x = self.brelu3(x)                               #64,256,256
        
        x = self.tconv4(x)
        x = self.brelu4(x)
        return x
        