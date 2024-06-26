import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from collections import OrderedDict


# 添加SaltLayer类
class SaltLayer(nn.Module):
    def __init__(self, salt_dim):
        super(SaltLayer, self).__init__()
        self.salt_dim = salt_dim

    def forward(self, x):
        salt = torch.randn_like(x[:, :self.salt_dim])  # 生成与输入相同维度的噪声
        return torch.cat((x, salt), dim=1)  # 将噪声与输入拼接

    def decode(self, x):
        return x[:, :-self.salt_dim]  # 去除噪声，恢复原始输入

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNWithSalt(nn.Module):
    def __init__(self, num_classes=10, salt_dim=16):
        super(SimpleCNNWithSalt, self).__init__()
        self.salt_layer = SaltLayer(salt_dim)
        self.conv1 = nn.Conv2d(3 + salt_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.salt_layer(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SenFier_ML(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        cl_inp_size = 100
        fc_size = 128
        self.fcml1 = nn.Sequential(
            nn.Linear(in_features=cl_inp_size, out_features=fc_size),            
            nn.Dropout(p=0.2, inplace=True)
            )
        self.fcml2 = nn.Sequential(
            nn.Linear(in_features=fc_size, out_features=num_classes)
        )

    def forward(self, x):
        out = self.fcml1(x)
        out = self.fcml2(out)        
        return out


class SenNet(nn.Module):
    def __init__(self, salt_layer=-1,                 
                classifier=SenFier_ML(num_classes=13)):
        super(SenNet, self).__init__()        
        
        self.salt_layer = salt_layer
        slat_layer_channels = 0
        
        slat_layer_channels = [0,0,0]
        if salt_layer == 2:
            slat_layer_channels[0] = 4
            self.salt_inp = nn.ConvTranspose2d(1, 4, (27,96), stride=4)                            
        elif salt_layer == 3:
            slat_layer_channels[1] = 16
            self.salt_inp = nn.ConvTranspose2d(1, 16, (27,46), stride=4)                            
        elif salt_layer == 4:
            slat_layer_channels[2] = 8
            self.salt_inp = nn.ConvTranspose2d(1, 8, (13,21), stride=4)                            

        self.num_chan = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,1)),            
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),            
            # nn.MaxPool2d(kernel_size=(1, 2), stride=(1,2)),
        )        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan+slat_layer_channels[0], out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan+slat_layer_channels[1], out_channels=self.num_chan, kernel_size=(3, 5), stride=(2,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan+slat_layer_channels[2], out_channels=self.num_chan, kernel_size=(3, 5), stride=(2,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=100, kernel_size=(1, 5), stride=(2,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=100)),
            nn.LeakyReLU(),            
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3)), nn.Dropout(0.5)
        )
        
        self.classifier = classifier

    def forward(self, x, salt=None):
        
        out = self.conv1(x)        
        out = self.conv2(out)  

        if self.salt_layer ==2:
            salt = self.salt_inp(salt)            
            out = torch.cat([out, salt], 1)

        out = self.conv3(out)

        if self.salt_layer ==3:
            salt = self.salt_inp(salt)            
            out = torch.cat([out, salt], 1)

        out = self.conv4(out)        
        
        if self.salt_layer ==4:
            salt = self.salt_inp(salt)            
            out = torch.cat([out, salt], 1)

        out = self.conv5(out)
        out = self.conv6(out)
        embed = self.pool(out)

        embed = embed.reshape(-1, 100 * 1)
        out = self.classifier(embed)
        return out

# 添加一个具有编码和解码层的类
class SimpleCNNwithEncoderDecoder(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Decoder part
        self.decoder_fc = nn.Linear(256, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        fc_output = self.fc1(encoded)
        
        decoded = self.decoder_fc(fc_output)
        decoded = decoded.view(-1, 128, 8, 8)
        decoded = self.decoder(decoded)
        
        output = self.fc2(fc_output)
        
        return output, decoded

def simplecnnwithEncoderDecoder(num_classes=10):
    return SimpleCNNwithEncoderDecoder(num_classes)

class SimpleCNN(nn.Module):
    def __init__(self,  num_classes=10, salt_layer=-1, 
                mean: Union[Tuple[float, ...], float] = None,
                std: Union[Tuple[float, ...], float] = None,
                num_input_channels: int = 3
                ):
        super().__init__()
        
        self.salt_layer = salt_layer
        slat_layer_channels = [0,0,0,0,0,0,0,0,0]

        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        
        if salt_layer == 0:                
            slat_layer_channels[0] = 1
            self.salt_inp = nn.ConvTranspose2d(1, 1, 32, stride=4)
        elif salt_layer == 1:
            slat_layer_channels[1] = 5
            self.salt_inp = nn.ConvTranspose2d(1, 5, 28, stride=4)
        elif salt_layer == 2:                         
            self.salt_inp = nn.ConvTranspose2d(1, 5, 24, stride=4)
        elif salt_layer == 3:                
            slat_layer_channels[3] = 5       
            self.salt_inp = nn.ConvTranspose2d(1, 5, 12, stride=4)
        elif salt_layer == 4:                                
            self.salt_inp = nn.ConvTranspose2d(1, 5, 8, stride=4)
        elif salt_layer == 5:                
            slat_layer_channels[5] = 8       
            self.salt_inp = nn.ConvTranspose2d(1, 8, 4, stride=4)
        elif salt_layer == 6:                
            slat_layer_channels[6] = num_classes
        elif salt_layer == 7:                
            slat_layer_channels[7] = num_classes
        elif salt_layer == 8:                
            slat_layer_channels[8] = num_classes
        else:
            pass                
        
        self.conv_1 = nn.Conv2d(num_input_channels+slat_layer_channels[0], 32, 5)        
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv2d(32+slat_layer_channels[1], 32, 5)
        self.relu_2 = nn.ReLU()

        self.pool_2 = nn.MaxPool2d(2, 2)        

        self.conv_3 = nn.Conv2d(32+slat_layer_channels[3], 32, 5)
        self.relu_3 = nn.ReLU()

        self.pool_3 = nn.MaxPool2d(2, 2)        

        self.flatten = nn.Flatten()

        self.fc_1 = nn.Linear((32+slat_layer_channels[5]) * 4 * 4 + slat_layer_channels[6], 120)        
        self.relu_3 = nn.ReLU()

        self.fc_2 = nn.Linear(120+slat_layer_channels[7], 84)
        self.relu_4 = nn.ReLU()

        self.fc_3 = nn.Linear(84+slat_layer_channels[8], num_classes)


    def forward(self, x, salt=None):
        
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std        

        if 0<= self.salt_layer <=5:
            salt = self.salt_inp(salt)

        if self.salt_layer == 0:    
            x = torch.cat([x, salt], 1)    
        x = self.relu_1(self.conv_1(x))
        
        if self.salt_layer == 1:    
            x = torch.cat([x, salt], 1)
        x = self.relu_2(self.conv_2(x))
        
        if self.salt_layer == 2:    
            x = torch.cat([x, salt], 1)
        x = self.pool_2(x)

        if self.salt_layer == 3:    
            x = torch.cat([x, salt], 1)
        x = self.relu_3(self.conv_3(x))
        
        if self.salt_layer == 4:    
            x = torch.cat([x, salt], 1)
        x = self.pool_3(x)

        if self.salt_layer == 5:    
            x = torch.cat([x, salt], 1)
        x = self.flatten(x) 
        
        if self.salt_layer == 6:    
            x = torch.cat([x, salt], 1)
        x = self.relu_3(self.fc_1(x))        

        if self.salt_layer == 7:    
            x = torch.cat([x, salt], 1)
        x = self.relu_4(self.fc_2(x))

        if self.salt_layer == 8:    
            x = torch.cat([x, salt], 1)
        x = self.fc_3(x)
        
        return x
