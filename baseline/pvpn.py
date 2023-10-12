# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

"""
    Implementation of https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8951138
    CNN Autoencoder 
"""

import math
import os
import yaml
import time
import argparse
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


from data.dataset import Dataset
from data.plot import draw_grid 
from utils import init_torch_seeds, save_ckp, load_checkpoint
from config import combine_cfgs
from GAN.visualize import draw_curve
from GAN.train import smooth_predictions
from GAN.loss import FocalLoss
from GAN.lr import get_scheduler

from metrics.metrics import Metrics, tabulate_runs


#### Xceptio-pytorhc ####
# https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
    
#### Xceptio-pytorhc ####
# https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=False)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


#### Xceptio-pytorhc ####
# https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self):
        """ Constructor
        """
        super(Xception, self).__init__()

        ## 1. Entry Flow ## 

        # Conv32 3x3, Stride=2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)

        # Conv64, 3x3, Stride=1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        # SeperableConv128, 3x3
        # SeperableConv128, 3x3
        # MaxPooling, 3x3, stride=2
        self.block1 = Block(in_filters=64, out_filters=128, reps=2, strides=2, start_with_relu=False, grow_first=True)
      
        # SeperableConv256, 3x3
        # SeperableConv256, 3x3
        # MaxPooling, 3x3, stride=2
        self.block2 = Block(in_filters=128, out_filters=256, reps=2, strides=2, start_with_relu=True, grow_first=True)

        # SeperableConv728, 3x3
        # SeperableConv728, 3x3
        # MaxPooling, 3x3, stride=2
        self.block3 = Block(in_filters=256, out_filters=728, reps=2, strides=2, start_with_relu=True, grow_first=True)

        ## 2. Middle Flow ## 

        # SeperableConv728, 3x3
        # SeperableConv728, 3x3
        # SeperableConv728, 3x3
        self.block4 = Block(in_filters=728, out_filters=728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        
        # Repeat for 12 times!
        self.block5 = Block(in_filters=728, out_filters=728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block6 = Block(in_filters=728, out_filters=728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block7 = Block(in_filters=728, out_filters=728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block8 = Block(in_filters=728, out_filters=728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block9 = Block(in_filters=728, out_filters=728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block10 = Block(in_filters=728, out_filters=728, reps=3, strides=1, start_with_relu=True, grow_first=True)

        ## 3. Exit Flow ## 

        # SeperableConv728, 3x3
        self.block11 = SeparableConv2d(in_channels=728, out_channels=728, kernel_size=3, stride=1, padding=1)
        self.block11_bn =  nn.BatchNorm2d(728)


        # SeperableConv1024, 3x3
        # MaxPooling, 3x3, stride=2
        self.block12 = SeparableConv2d(in_channels=728, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.block12_bn =  nn.BatchNorm2d(1024)

        self.block12_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.skip = nn.Conv2d(in_channels=728, out_channels=1024, kernel_size=1, stride=1, bias=False)
        self.skipbn = nn.BatchNorm2d(1024)

        # SeperableConv1536, 3x3
        self.conv3 = SeparableConv2d(in_channels=1024, out_channels=1536, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)

        # SeperableConv2048, 3x3
        self.conv4 = SeparableConv2d(in_channels=1536, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)


        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        ## 1. Entry Flow ##

        h1 = torch.clone(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        h2 = torch.clone(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        h3 = torch.clone(x)

        x = self.block1(x)
    
        h4 = torch.clone(x)
        
        x = self.block2(x)
        
        ## 2. Middle Flow ##
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
       
        ## 3. Exit Flow ##
        skip = self.skip(x)
        skip = self.skipbn(skip)
        skip = self.relu(skip)

        x = self.block11(x)
        x = self.block11_bn(x)
        x = self.relu(x)

        x = self.block12(x)
        x = self.block12_bn(x)
        x = self.relu(x)

        x = self.block12_maxpool(x)
        x += skip 

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        return x, h1, h2, h3, h4


#### Xceptio-pytorhc ####
# https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
       
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


### PVPN Implementation ### 
class Decoder(nn.Module):

    def __init__(self, D=1):
        super(Decoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1.95), mode='bilinear'),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=(2.1, 2), mode='bilinear'),
            nn.Conv2d(256, 256 // 2, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='bilinear'),
            nn.Conv2d(256 // 2, 256 // 4, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 1), mode='bilinear'),
            nn.Conv2d(256 // 4, 256 // 8, 3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.block5 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='bilinear'),
            nn.Conv2d(256 // 8, D, 3, stride=1, padding=1),
        )
      
    
    def forward(self, enc_out, h4, h3, h2, h1):
        out1 = self.block1(enc_out) 

        out2 = self.block2(out1)
        h4 = F.pad(input=h4, pad=(0, 1, 0, 1), mode='constant', value=0)
        skip = out2 + h4 

        out3 = self.block3(out2)
        h3 = F.pad(input=h3, pad=(0, 3, 0, 3), mode='constant', value=0)
        skip = out3 + h3 

        out4 = self.block4(skip)
        h2 = F.pad(input=h2, pad=(0, 1, 0, 1), mode='constant', value=0)
        skip = out4 + h2

        out5 = self.block5(skip)
        out5 += h1   
        out5 = nn.Sigmoid()(out5)

        return out5 


class PVPN(nn.Module):

    def __init__(self, num_measurements, mesh_height, mesh_width, device, D=1):
        super(PVPN, self).__init__()

        mesh_res = mesh_height*mesh_width

        self.inv = nn.Sequential(
            nn.Linear(num_measurements, mesh_res // 2),
            nn.Sigmoid(),
            nn.Linear(mesh_res // 2, mesh_res),
            nn.Sigmoid()
        ) 

        self.fn = nn.Sequential(
            nn.Linear(mesh_res, mesh_res // 2),
            nn.Sigmoid(),
            nn.Linear(mesh_res // 2, num_measurements),
            nn.Sigmoid()
        )

        self.encoder = Xception()

        self.aspp = ASPP(backbone="", output_stride=8, BatchNorm=nn.BatchNorm2d)

        self.decoder = Decoder()

        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.device = device
    
    def forward(self, measurements):
        # inverse network
        inv_out = self.inv(measurements)
        
        # forward network
        fn_out = self.fn(inv_out)
        
        # reshape inv_out
        pred_perm = inv_out.reshape((-1, 1, self.mesh_height, self.mesh_width))
        
        # encoder 
        enc_out, h1, h2, h3, h4 = self.encoder(pred_perm)

        # ASPP
        aspp_out = self.aspp(enc_out)

        # decoder
        dec_out = self.decoder(aspp_out, h4, h3, h2, h1)
       
        return inv_out, fn_out, dec_out 
    
    
    def loss(self, measurement, permittivity, inv_out, fn_out, dec_out):        
        weights = [0.25, 0.25, 0.5]
        
        l1_loss = nn.SmoothL1Loss() 
        focal_loss = FocalLoss(device=self.device, gamma=2, alpha=0.5)
       
        perm_flattened = permittivity.reshape((-1, self.mesh_height*self.mesh_width))
        loss = weights[0] * l1_loss(inv_out, perm_flattened) + weights[1] * l1_loss(fn_out, measurement) + weights[2] * focal_loss(dec_out, permittivity)
        
        return loss


def train_one_epoch(model, train_loader, val_loader, optimizer, epoch, device):
    train_losses = 0
    val_losses = 0 

    for x, y in train_loader: 
        perm  = x["perm_xy"]
        vb = y["v_b"]

        perm = perm.to(device)
        vb = vb.to(device)

        optimizer.zero_grad()

        model.train(True)

        input_g = vb.view(vb.shape[0], vb.shape[1])

        inv_out, fn_out, dec_out  = model(input_g.float())
       
        loss = model.loss(vb.float(), perm.float(), inv_out, fn_out, dec_out)

        loss.backward()
        optimizer.step()

        train_losses += loss.item()


    model.train(False)

    for x, y in val_loader:
        perm  = x["perm_xy"]
        vb = y["v_b"]

        perm = perm.to(device)
        vb = vb.to(device)

        input_g = vb.view(vb.shape[0], vb.shape[1])
        inv_out, fn_out, dec_out  = model(input_g.float())

        loss = model.loss(vb.float(), perm.float(), inv_out, fn_out, dec_out)
        val_losses += loss.item() 


    train_avg_loss = train_losses / len(train_loader)
    val_avg_loss = val_losses / len(val_loader)

    print('Epoch: %0.2f | Training Loss: %.6f | Validation Loss: %0.6f'  % (epoch, train_avg_loss, val_avg_loss))

    return float(train_avg_loss), val_avg_loss 


def test(model, test_loader, config, output_dir, device):
    run_time = 0.0
    predictions = torch.tensor([], device="cpu")
    ground_truth = torch.tensor([], device="cpu")

    model.train(False) 

    pred_path = os.path.join(output_dir, "pred")
    truth_path = os.path.join(output_dir, "truth")
    
    os.makedirs(pred_path, exist_ok=True)
    os.makedirs(truth_path, exist_ok=True)

    for i, (x, y) in enumerate(tqdm(test_loader)):
        perm  = x["perm_xy"].float()
        vb = y["v_b"]

        perm = perm.to(device)
        vb = vb.to(device)

        input_g = vb.view(vb.shape[0], vb.shape[1])

        st = time.time()
        _, _, dec_out  = model(input_g.float())
        run_time += time.time() - st 

        # smooth predictions
        predicted_perm = dec_out.reshape((-1, 1, 100, 200))
        pred_perm_smoothed, _ = smooth_predictions(predicted_perm, torch.tensor([]), config.MODEL.HEAD_ACTIVATION, config.DATASET.POS_VALUE, config.DATASET.NEG_VALUE)
        # pred_perm_smoothed = predicted_perm
        
        predictions = torch.cat((predictions, pred_perm_smoothed.cpu().detach()))
        ground_truth = torch.cat((ground_truth, perm.cpu().cpu().detach()))

        for j, pred in enumerate(pred_perm_smoothed): 
            pred_perm = pred
            draw_grid(pred_perm[0].cpu().detach().numpy(), "ECT Prediction", "Row (200\u03bcm)", "Depth (100\u03bcm)", colorbar=False, font_size=18, save_path=os.path.join(pred_path, f"pred_{i}_{j}.png"))
            draw_grid(pred_perm[0].cpu().detach().numpy(), "ECT Prediction", "Row (y)", "Depth (z)", colorbar=False, ticks=False, scale_bar=True, font_size=24, save_path=os.path.join(pred_path, f"pred_{i}_{j}.png"))

            draw_grid(perm[j][0].cpu().detach().numpy(), "Confocal Microscopy", "Row (200\u03bcm)", "Depth (100\u03bcm)", colorbar=False, font_size=18, save_path=os.path.join(truth_path, f"truth_{i}_{j}.png"))
            draw_grid(perm[j][0].cpu().detach().numpy(), "Confocal Microscopy", "Row (y)", "Depth (z)", colorbar=False,  ticks=False, scale_bar=True, font_size=24, save_path=os.path.join(truth_path, f"truth_{i}_{j}.png"))

    run_time = run_time / len(test_loader)

    return  predictions, ground_truth, run_time



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to training configuration.", required=True)
    parser.add_argument('--checkpoint', type=str, help="Path to pretrained model.", required=False)
    parser.add_argument('--best_model', type=str, help="Path to best model [For Testing].", required=False)
    parser.add_argument('--output', type=str, help="Path to output directory.", default="synthetic/baseline")  

    args = parser.parse_args()

    config_path = args.config
    checkpoint = args.checkpoint 
    best_model = args.best_model
    output_dir = args.output 

    if best_model is None: 
        best_model = os.path.join(output_dir, "best_model.pth")

    config = combine_cfgs(config_path)
    
    seed = config.SEED 
    batch_size = config.DATASET.BATCH_SIZE
    data_path = config.DATASET.PATH
    num_measurements = config.DATASET.NUM_MEASUREMENTS 
    normalize = config.DATASET.NORMALIZE 
    shuffle = config.DATASET.SHUFFLE
    standardize = config.DATASET.STANDARDIZE 
    noise = config.DATASET.NOISE 
    noise_stdv = config.DATASET.NOISE_STDV
    pos_value = config.DATASET.POS_VALUE
    neg_value = config.DATASET.NEG_VALUE
    lr = config.SOLVER.LEARNING_RATE
    lr_scheduler = config.SOLVER.LR_SCHEDULER
    epochs = config.SOLVER.EPOCHS
    train_split, val_split, _ = config.DATASET.TRAIN_VAL_TEST_SPLIT    

    os.makedirs(output_dir, exist_ok=True)

    init_torch_seeds(seed)

    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    dataset = Dataset(data_path, shuffle=shuffle, normalize=normalize, standardize=standardize, pos_value=pos_value, neg_value=neg_value, device=device)
    
    train_length = int(len(dataset)*train_split)
    val_length = int((len(dataset)*val_split))
    test_length = int((len(dataset) - train_length - val_length))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, val_length, test_length], generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True) 

    # Load pretrained model state 
    start_epoch = 0
    if checkpoint:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint)

    # Prepare model 
    model = PVPN(num_measurements=num_measurements, mesh_height=100, mesh_width=200, device=device)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Parameters: ", num_params)
    
    # summary(model, [(1, num_measurements), (1, 1, 100, 200)])
    # model.apply(weights_init)

    min_valid_loss = 1_000_000
    train_loss = []
    val_loss = []
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = get_scheduler(lr_scheduler, optimizer, T=epochs)

    # Training Loop
    for i in range(start_epoch, start_epoch+epochs):

        train_avg_loss, val_avg_loss = train_one_epoch(model, train_loader, val_loader, optimizer, i, device)
        scheduler.step()

        writer.add_scalar("Loss/train", train_avg_loss, i)
        writer.add_scalar("Loss/val", val_avg_loss, i)

        if min_valid_loss > val_avg_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f} ---> {val_avg_loss:.6f}) \t Saving The Model')
           
            # Saving State Dict
            checkpoint = {'epoch': i + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            
            # save_ckp is very slow
            # save_ckp(checkpoint, is_best=True, checkpoint_dir=output_dir, best_model_path=os.path.join(output_dir, "best_model.pth"))
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

            min_valid_loss = val_avg_loss

        train_loss.append(train_avg_loss)
        val_loss.append(val_avg_loss)
        
        draw_curve(i, train_loss, val_loss, "FL+L1Smooth+L1Smooth", os.path.join(output_dir, ))

        # save_ckp is very slow
        # # save checkpoint every 50 epochs 
        # if i % 50 == 0: 
        #     checkpoint = {
        #         'epoch': i + 1,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict()
        #     }
        #     save_ckp(checkpoint, is_best=False, checkpoint_dir=output_dir, best_model_path=None)

    # Testing Loop
    # model, _, _ = load_checkpoint(model, optimizer, os.path.join(output_dir, "best_model.pth"))
    # model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    model.load_state_dict(torch.load(best_model))

    predictions, ground_truth, run_time = test(model, test_loader, config, output_dir, device)

    predictions = predictions.to(device)
    ground_truth = ground_truth.to(device)
    
    metrics = Metrics(device=device)
    metrics = metrics.forward(predictions, ground_truth)
    print(metrics)

    stats, table = tabulate_runs([metrics], run_time, os.path.join(output_dir, "stats.json"))
    print(table.draw())
    
    writer.add_scalar("SSIM_acc", metrics["SSIM"])
    writer.add_scalar("MSE_acc", metrics["MSE"])
    writer.add_scalar("MAE_acc", metrics["MAE"])
    writer.add_scalar("PSNR_acc", metrics["PSNR"])
    writer.add_scalar("IoU_acc", metrics["IoU"])
    writer.add_scalar("CC_acc", metrics["CC"])

    writer.flush()
    writer.close()



if __name__ == "__main__":
    main()


