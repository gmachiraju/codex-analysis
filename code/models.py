import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.models as models
from torchvision import transforms as trn
from torchsummary import summary
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



#new loss: adapted from: https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681
class MultiTaskLoss(nn.Module):
	def __init__(self, model, eta, combo_flag):
		super(MultiTaskLoss, self).__init__()
		self.combo_flag = combo_flag
		self.model = model
		self.eta = nn.Parameter(torch.tensor(eta, dtype=dtype))

	def forward(self, inputs, y, loss2):
		scores = self.model(inputs)
		loss1 = F.cross_entropy(scores, y, reduction="mean")
		losses = [loss1, loss2]

		if self.combo_flag == "uncertainty":
			total_l = torch.tensor(losses) * torch.exp(-self.eta) + self.eta
			total_loss = total_l.sum()
		elif self.combo_flag == "learnAlpha":
			total_loss = torch.tensor(loss1, dtype=dtype) + (self.eta * torch.tensor(loss2, dtype=dtype))

		return scores, losses, total_loss


# Non-neural torch modules
#--------------------------

class VGGEmbedder(torch.nn.Module):
	def __init__(self, trained_model, args, att_flag=False):
		super(VGGEmbedder, self).__init__()
		# if att_flag == True:
		#     self.embedder = trained_model[:-3].eval()
		if args.model_class in ["VGG19", "VGG19_bn"]:
			if args.patch_size == 224:
				self.embedder = trained_model.eval()
				#torch.nn.Sequential(*(list(trained_model.children()))).eval()
				# self.embedder = torch.nn.Sequential(*(list(trained_model.children())[:-1])).eval()

		# self.embedder = nn.Sequential(*list(trained_model.classifier.children())[:-3]).eval() 
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
        self.output_layer = torch.nn.Linear(n_inputs, 1)
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
class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
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


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1  # used to be 3
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


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
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


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)




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
