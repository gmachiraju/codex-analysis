import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.models as models
from torchvision import transforms as trn
from torchsummary import summary
from torch.autograd import Variable
import pytorch_lightning as pl

import numpy as np
import pdb
import copy
import gc
from typing import Union, List, Dict, Any, cast

LEARN_RATE_IMG = 8e-2 #5e-2 #1e-2
dtype=torch.float32

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


def flatten(x):
    N = x.shape[0] # read in N, C, H, W, where N is batch size
    return x.view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
    
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x

# ConvMixer
# def ConvMixer(h, depth, kernel_size=9, patch_size=7, n_classes=2):
#     Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))
#     Residual = type('Residual', (Seq,), {'forward': lambda self, x: self[0](x) + x})
#     return Seq(ActBn(nn.Conv2d(3, h, patch_size, stride=patch_size)),
#             *[Seq(Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))),
#             ActBn(nn.Conv2d(h, h, 1))) for i in range(depth)],
#             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(h, n_classes))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, in_channels, kernel_size=9, patch_size=7, n_classes=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


#-------------------tile2vec--------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# # adapted from tile2vec: https://github.com/ermongroup/tile2vec/blob/master/src/tilenet.py
# class ResNetHighD(nn.Module):
#     def __init__(self, num_blocks, in_channels=3, z_dim=512):
#         super(ResSelfEmbedder, self).__init__()
#         self.in_channels = in_channels
#         self.z_dim = z_dim
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1,
#             padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
#         self.layer5 = self._make_layer(self.z_dim, num_blocks[4], stride=2)

#     def _make_layer(self, planes, num_blocks, stride, no_relu=False):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def encode(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = F.avg_pool2d(x, 4) # this was the problem area for high-D inputs
#         z = x.view(x.size(0), -1)
#         return z

#     def forward(self, x):
#         return self.encode(x)

#     def triplet_loss(self, z_p, z_n, z_d, margin=10, l2=0):
#         pdb.set_trace()
#         l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
#         l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
#         l_nd = l_n + l_d
#         loss = F.relu(l_n + l_d + margin)
#         l_n = torch.mean(l_n)
#         l_d = torch.mean(l_d)
#         l_nd = torch.mean(l_n + l_d)
#         loss = torch.mean(loss)
#         if l2 != 0:
#             loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
#         return loss, l_n, l_d, l_nd

#     def loss(self, patch, neighbor, distant, margin=10, l2=0):
#         """
#         Computes loss for each batch.
#         """
#         z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor), self.encode(distant))
#         return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, no_relu=False,
        activation='relu'):
        super(BasicBlock, self).__init__()
        self.no_relu = no_relu
        self.activation = activation

        # Choose activation function
        if self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # no_relu layer
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        # no_relu layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.activation_fn(self.bn1(self.conv1(x)))
        if self.no_relu:
            out = self.bn3(self.conv3(out))
            return out
        else:
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            # out = F.relu(out)
            out = self.activation_fn(out)
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_classes=10, in_channels=3, z_dim=512, supervised=False, no_relu=False, loss_type='triplet', tile_size=224, activation='relu'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.supervised = supervised
        self.no_relu = no_relu
        self.loss_type = loss_type
        self.tile_size = tile_size
        self.activation = activation

        # Choose activation function
        if self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, self.z_dim, num_blocks[4], stride=2, no_relu=self.no_relu)
        self.linear = nn.Linear(self.z_dim*block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, no_relu=no_relu, activation=self.activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x, verbose=False):
        # x = F.relu(self.bn1(self.conv1(x)))
        x = self.activation_fn(self.bn1(self.conv1(x)))
        if verbose: print(x.shape)
        x = self.layer1(x)
        if verbose: print(x.shape)
        x = self.layer2(x)
        if verbose: print(x.shape)
        x = self.layer3(x)
        if verbose: print(x.shape)
        x = self.layer4(x)
        if verbose: print(x.shape)
        x = self.layer5(x)
        if verbose: print(x.shape)
        
        if self.tile_size == 50:
            x = F.avg_pool2d(x, 4)
        elif self.tile_size == 25:
            x = F.avg_pool2d(x, 2)
        elif self.tile_size == 75:
            x = F.avg_pool2d(x, 5)
        elif self.tile_size == 100:
            x = F.avg_pool2d(x, 7)
        elif self.tile_size == 224: 
            # added this for larger inputs
            x = F.avg_pool2d(x, 14)

        if verbose: print('Pooling:', x.shape)
        z = x.view(x.size(0), -1)
        if verbose: print('View:', z.shape)
        return z

    def forward(self, x):
        if self.supervised:
            z = self.encode(x)
            return self.linear(z)
        else:
            return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=10, l2=0.01):
        # pdb.set_trace()
        """
        z_i = [B,d] 
            B: batch size
            d: hidden dim
        """
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)

        l_nd = torch.mean(l_nd)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        # l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def sextuplet_loss(self, p0, n0, d0, p1, n1, d1, margin=10):
        z_p0, z_n0, z_d0 = (self.encode(p0), self.encode(n0), self.encode(d0))
        z_p1, z_n1, z_d1 = (self.encode(p1), self.encode(n1), self.encode(d1))
        centroid0 = torch.mean(torch.stack([z_p0, z_n0, z_d0]), dim=0)
        centroid1 = torch.mean(torch.stack([z_p1, z_n1, z_d1]), dim=0)
        l = - torch.sqrt(((centroid0 - centroid1) ** 2).sum(dim=1))
        loss = F.relu(l + margin)
        loss = torch.mean(loss)
        return loss

    def loss(self, patch, neighbor, distant, margin=10, l2=0, verbose=False):
        """
        Computes loss for each batch.
        """            
        z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor), self.encode(distant))
        if verbose == True:
            print("embed shape:", z_p.shape)
        if self.loss_type == 'triplet':
            return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)
        elif self.loss_type == 'cosine':
            return self.cosine_loss(z_p, z_n, z_d)


def ResNet18(n_classes=10, in_channels=3, z_dim=512, supervised=False, no_relu=False, loss_type='triplet', tile_size=224, activation='relu'):
    return ResNet(BasicBlock, [2,2,2,2,2], n_classes=n_classes, in_channels=in_channels, z_dim=z_dim, supervised=supervised, no_relu=no_relu, loss_type=loss_type, tile_size=tile_size, activation=activation)
#-----------------------------------------------------------


#new loss: adapted from: https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681
class MultiTaskLoss(nn.Module):
    def __init__(self, model, eta, combo_flag):
        super(MultiTaskLoss, self).__init__()
        self.combo_flag = combo_flag
        self.model = model
        self.eta = nn.Parameter(torch.tensor(eta, dtype=dtype))
        # self.args = args
        # if args.patch_size == 224 and args.model_class == "VGG19_bn":
        # 	pass
        # else:
        # 	self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, inputs, y, loss2, device=torch.device("cuda")):
        
        scores = self.model(inputs)
        # loss1 = F.binary_cross_entropy_with_logits(scores, y, reduction="mean")
        loss1 = F.cross_entropy(scores, y, reduction="mean")
        # loss1 = self.loss(scores, y, reduction=mean)
        losses = [loss1, loss2]
        
        if self.combo_flag == "uncertainty":
            total_l = torch.tensor(losses, device=device) * torch.exp(-self.eta) + self.eta
            total_loss = total_l.sum()
        elif self.combo_flag == "learnAlpha":
            total_loss = torch.tensor(loss1, dtype=dtype) + (self.eta * torch.tensor(loss2, dtype=dtype))

        return scores, losses, total_loss


# Non-neural torch modules
#--------------------------

class VGGEmbedder(torch.nn.Module):
	def __init__(self, trained_model, args, att_flag=False):
		super(VGGEmbedder, self).__init__()
		if args.model_class in ["VGG19", "VGG19_bn"]:
			if args.patch_size == 224:
				if args.backprop_level == "none":
					# pdb.set_trace()
					# self.embedder = torch.nn.Sequential(*(list(trained_model.children())[:-1])).eval()
					# self.embedder = torch.nn.Sequential(*(list(trained_model.children())[:-1])).eval()
					# self.embedder = torch.nn.Sequential(*(list(trained_model.children())[:])).eval()
					
					# reassigning changes the model, copying is not beneficial to the optimization
					clf = torch.nn.Sequential(trained_model.classifier[0])
					copied_model = copy.deepcopy(trained_model)
					copied_model.classifier = clf
					self.embedder = copied_model.eval()					
				elif args.backprop_level == "blindfolded": #  blindfolded backprop
					# clf = torch.nn.Sequential(trained_model.classifier[0])
					# trained_model.classifier = clf
					# self.embedder = trained_model # keep gradient flow going

					# use hooks!
					# pdb.set_trace()
					self.embedder = trained_model.eval()
				else: # full
					print("full backprop perserved")
					self.embedder = trained_model
		else:
			print("Error: Have not yet implemented VGG-Att embeddings!")
			exit()
	
	def forward(self, x):
	    return self.embedder(x)


class ViTEmbedder(torch.nn.Module):
    def __init__(self, trained_model, args):
        super(ViTEmbedder, self).__init__()
        self.embedder = trained_model[:-3].eval()

    def forward(self, x):
        return self.embedder(x)
        

# Model: https://www.kaggle.com/negation/pytorch-logistic-regression-tutorial
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
        # self.sigmoid = torch.nn.Sigmoid(num_classes) # added
    
    def forward(self, x):
        out = torch.sigmoid(self.linear(x)) # self.sigmoid(self.linear(x)) 
        return out


# https://www.richard-stanton.com/2021/06/19/pytorch-elasticnet.html
class ElasticLinear(pl.LightningModule):
    def __init__(self, loss_fn, n_inputs: int = 1, learning_rate=LEARN_RATE_IMG, l1_lambda=0.05, l2_lambda=0.05):
        super().__init__()
        # learning_rate used to be 0.05
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(n_inputs, 2)
        self.train_log = []

    def forward(self, x):
        outputs = self.output_layer(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()
        
        return self.l2_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # ensure type/shape for y_hat, y -- floats work for both
        y = torch.squeeze(y) # size: batch
        y = y.type(torch.LongTensor)
        y_hat = y_hat.type(torch.FloatTensor)

        loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()
        
        self.log("loss", loss)
        self.train_log.append(loss.detach().numpy())
        return loss


# from: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
class Seq_Ex_Block(nn.Module):
    def __init__(self, in_ch, r):
        super(Seq_Ex_Block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//r, in_ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        #print(f'x:{x.sum()}, x_se:{x.mul(se_weight).sum()}')
        return x.mul(se_weight)


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self, x):
        return x.view(*(x.shape[:-2]),-1).mean(-1)


class GlobalMaxPool(nn.Module):
    def __init__(self):
        super(GlobalMaxPool, self).__init__()
    def forward(self, x):
        return x.view(*(x.shape[:-2]),-1).max(-1)


# Architectures
#---------------

class VGG19():
	# same structure as vgg19 model, modified inputs and outputs
	# print(torchvision.models.vgg19(pretrained=False, progress = True))
	def __init__(self, bn_flag=False, D=1):
		self.bn_flag = bn_flag
	
		if bn_flag == False:
			self.arch = nn.Sequential(
		    
			    nn.Conv2d(D, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #64
			    nn.ReLU(inplace = True),
			    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    Flatten(),
			    
			    nn.Linear(in_features=4608, out_features=4096, bias=True),
			    nn.ReLU(inplace = True),
			    nn.Dropout(p=0.5),
			    
			    nn.Linear(in_features=4096, out_features=4096, bias=True),
			    nn.ReLU(inplace = True),
			    nn.Dropout(p=0.5),
			    
			    nn.Linear(in_features=4096, out_features=2, bias=True),
			)

		elif bn_flag == True:
			# same structure as vgg19 model, modified inputs and outputs
			# print(torchvision.models.vgg19(pretrained=False, progress = True))
			self.arch = nn.Sequential(
			    
			    nn.Conv2d(D, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #64
			    nn.BatchNorm2d(128),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(128),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(256),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(256),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(256),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(256),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(256),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(256),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(512),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(512),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(512),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(512),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(512),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(512),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(512),
			    nn.ReLU(inplace = True),
			    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			    nn.BatchNorm2d(512),
			    nn.ReLU(inplace = True),
			    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			    
			    Flatten(),
			    
			    nn.Linear(in_features=4608, out_features=4096, bias=True),
			    nn.ReLU(inplace = True),
			    nn.Dropout(p=0.5),
			    
			    nn.Linear(in_features=4096, out_features=4096, bias=True),
			    nn.ReLU(inplace = True),
			    nn.Dropout(p=0.5),
			    
			    nn.Linear(in_features=4096, out_features=2, bias=True),
			)


        
class stdVGG19():
	def __init__(self, bn_flag=False):

		# VGG19 portion, pre-pooling
		if bn_flag == False:
			self.arch = torchvision.models.vgg19() # pretrained=False
		else:
			self.arch = torchvision.models.vgg19_bn()

		self.arch.classifier[6] = nn.Linear(4096, 2) # assigning last layer to only 2 outputs

		for param in self.arch.parameters():
			param.requires_grad = True


# model from: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
# originally, num_classes=1000
class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 2, init_weights: bool = True, dropout: float = 0.5) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
    layers: List[nn.Module] = []
    # in_channels = 3  # used to be 1, 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, in_channels: int = 3, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, in_channels, **kwargs)




#################################################################################################
# Learn to Pay Attention Model - visual attention with VGGs 
# Code from: https://github.com/SaoYan/LearnToPayAttention
################################################################################################

def weights_init_kaimingUniform(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, a=0, b=1)
            nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)


def weights_init_kaimingNormal(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)


def weights_init_xavierUniform(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, a=0, b=1)
            nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)


def weights_init_xavierNormal(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)


# Blocks
#--------
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)


class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    
    def forward(self, inputs):
        return self.op(inputs)


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

'''
Grid attention block
Reference papers
Attention-Gated Networks https://arxiv.org/abs/1804.05338 & https://arxiv.org/abs/1808.08114
Reference code
https://github.com/ozan-oktay/Attention-Gated-Networks
'''
class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C)
        return c.view(N,1,W,H), output


'''
attention before max-pooling
'''
class AttnVGG_before(nn.Module):
    def __init__(self, ch_dim, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnVGG_before, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(ch_dim, 64, 2) # used to have a 3 in front
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        l1 = self.conv_block3(x) # /1
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0) # /2
        l2 = self.conv_block4(x) # /2
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0) # /4
        l3 = self.conv_block5(x) # /4
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0) # /8
        x = self.conv_block6(x) # /32
        g = self.dense(x) # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            x = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]
