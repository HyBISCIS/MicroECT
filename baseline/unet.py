# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 


"""
    Implementation of https://www.sciencedirect.com/science/article/pii/S0955598622001030#sec5
    Transformer + UNet
"""

import os
import yaml
import time
import argparse
import math

import numpy as np 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary


from data.dataset import Dataset
from data.plot import draw_grid 
from GAN.visualize import draw_curve
from GAN.train import smooth_predictions
from utils import init_torch_seeds, save_ckp, load_checkpoint
from config import combine_cfgs

from metrics.metrics import Metrics, tabulate_runs

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

    
class ConvNeXt(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, out_h=None, out_w=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            SeparableConv2d(in_channels, mid_channels, kernel_size=(7, 7), padding=3, bias=False),
            nn.LayerNorm([mid_channels, out_h , out_w]), # replace with layer norm
            nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        out = self.double_conv(x)
        out += x # Residual connection
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, out_h, out_w):
        super().__init__()
        self.conv1 = ConvNeXt(in_c, in_c, out_h=out_h, out_w=out_w)
        self.conv2 = ConvNeXt(in_c, in_c, out_h=out_h, out_w=out_w)
        self.down = Down(in_channels=in_c, out_channels=out_c, out_h=out_h, out_w=out_w)
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        p = self.down(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv3x3_1 = nn.Conv2d(in_channels=in_c+in_c, out_channels=in_c, kernel_size=(3, 3), padding=(2, 2))
        self.activ1 = nn.GELU()
        self.conv3x3_2 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3))
        self.activ2 = nn.GELU()

    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x, skip], axis=1)    
        x = self.conv3x3_1(x)
        x = self.activ1(x)
        x = self.conv3x3_2(x)
        x = self.activ2(x)
        
        return x

##################
### UNET Parts ###
##################

# UNET Parts from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class Down(nn.Module):
    """Downscaling layer"""

    def __init__(self, in_channels, out_channels, out_h, out_w):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.LayerNorm([in_channels, out_h , out_w]),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=2, padding=0, bias=False)
        )

    def forward(self, x):
        return self.down_sample(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

##################
##################
   
    

class UNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        """ Encoder """
        self.e1 = EncoderBlock(in_channels, 128, out_w=200, out_h=100)
        self.e2 = EncoderBlock(128, 256, out_w=100, out_h=50)
        self.e3 = EncoderBlock(256, 256, out_w=50, out_h=25)
       
        """ Bottleneck """
        self.b1 = ConvNeXt(256, 256, out_w=25, out_h=12)
        self.b2 = ConvNeXt(256, 256, out_w=25, out_h=12)

        """ Decoder """
        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)
       
        """ Classifier """
        self.outputs = OutConv(32, 1)

        
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        
        """ Bottleneck """
        b1 = self.b1(p3)
        b2 = self.b2(b1)

        """ Decoder """      
        d1 = self.d1(b2, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        
        """ Classifier """
        outputs = self.outputs(d3)
        
        return outputs
    
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    
class Transformer(nn.Module):
    
    def __init__(self, max_measurement=1, num_measurements=28, d_model=112, d_hid=2048, nhead=4, nlayers=2, dropout=0.1) -> None:
        super(Transformer, self).__init__()

        # self.embed = torch.nn.Embedding(num_embeddings=max_measurement+1, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_len=num_measurements)
            
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, activation="gelu", dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # self.mesh_res = num_measurements*d_model
        # self.inv = nn.Sequential(
        #     nn.Linear(num_measurements, self.mesh_res // 2),
        #     nn.GELU(),
        #     nn.Linear(self.mesh_res // 2, self.mesh_res),
        #     nn.GELU()
        # ) 
        
        self.embed_layer = nn.Sequential(
            nn.Linear(num_measurements, num_measurements*d_model),
        )
        
        self.d_model = d_model 
        
    def forward(self, x):
        L, bs = x.shape 
        # Input is [seq_Len, batch_size]
        # embed = self.embed(x.long())     # Output is ``[seq_len, batch_size, embedding_dim]``
        x = x.reshape(bs, L)
        embed = self.embed_layer(x).reshape(L, bs, self.d_model)
        # print("Input: ", x.shape, "Embedding: ", embed.shape)
        # print("Input: ", x, "Embedding: ", embed)
        # Input is [seq_len, batch_size, embedding_dim]
        embed_pe = self.pe(embed) + embed # Output is [seq_len, batch_size, embedding_dim]
        output = self.transformer_encoder(embed_pe) # Output is [seq_len, batch_size, embedding_dim]
        output = output.reshape(bs, L, self.d_model)
        return output 
    
    
class TUNet(nn.Module):
    
    def __init__(self, embed_dim=112, num_measurements=60, max_measurement=1, output_dim=(100, 200)):
        super(TUNet, self).__init__()
        self.output_dim = output_dim 
        # Feature Extraction
        self.transformer = Transformer(max_measurement=max_measurement, num_measurements=num_measurements, d_model=embed_dim)
        
        # upsamples the transformer output to be the size of the image
        pad_dims = [np.abs(num_measurements - self.output_dim[0]), np.abs(embed_dim - self.output_dim[1])]
        self.conv1x1 = nn.Conv2d(1, out_channels=64, padding=(pad_dims[0] // 2, pad_dims[1]), kernel_size=1)
        
        # Image Reconstruction Netowrk
        self.UNet = UNet(in_channels=64)
        self.loss_fn = nn.MSELoss()
        
        
    def forward(self, x): 
        bs, L = x.shape
        
        # 1. Transformer
        x = x.reshape(L, bs)
        transformer_output = self.transformer(x)
       
        # reshape output to 2D feature map 
        bs, L, embed = transformer_output.shape                 
        transformer_output = transformer_output.reshape(bs, 1, L, embed)
        transformer_output = self.conv1x1(transformer_output)

        # pad transformer output 
        pad_dims = (self.output_dim[1]-transformer_output.shape[3], 0, self.output_dim[0]-transformer_output.shape[2], 0)
        transformer_output_padded =  F.pad(transformer_output, pad_dims, "constant", 0)  # effectively zero padding

        # feature extraction network
        # print("trans output: ", transformer_output.shape)
        output = self.UNet(transformer_output_padded)  #[bs, 64, 64, 64]
        
        return output 

    def loss(self, pred, target):
        loss = self.loss_fn(pred, target)
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

        pred_perm = model(input_g.float())
       
        loss = model.loss(pred_perm, perm.float())
        
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

        pred_perm = model(input_g.float())

        loss = model.loss(pred_perm, perm)
        val_losses += loss.item() 


    train_avg_loss = train_losses / len(train_loader)
    val_avg_loss = val_losses / len(val_loader)

    print('Epoch: %0.2f | Training Loss: %.6f | Validation Loss: %0.6f'  % (epoch, train_avg_loss, val_avg_loss), flush=True)

    return float(train_avg_loss), val_avg_loss 


def smooth_predictions(predicted_perm, activation, pos_value, neg_value):
    pred_perm = torch.clone(predicted_perm)
    
    if activation == 'Sigmoid':
        pred_perm[predicted_perm >= 0.6] = pos_value
        pred_perm[predicted_perm < 0.6] = neg_value

    return pred_perm


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
        predicted_perm  = model(input_g.float())
        run_time += time.time() - st 

        # smooth predictions
        pred_perm_smoothed = smooth_predictions(predicted_perm, config.MODEL.HEAD_ACTIVATION, config.DATASET.POS_VALUE, config.DATASET.NEG_VALUE)
        # pred_perm_smoothed = predicted_perm
        
        predictions = torch.cat((predictions, pred_perm_smoothed.cpu().detach()))
        ground_truth = torch.cat((ground_truth, perm.cpu().cpu().detach()))

        for j, pred in enumerate(pred_perm_smoothed): 
            pred_perm = pred
            draw_grid(pred_perm[0].cpu().detach().numpy(), "ECT Prediction", xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)", font_size=18, colorbar=False, save_path=os.path.join(pred_path, f"pred_{i}_{j}.png"))
            draw_grid(pred_perm[0].cpu().detach().numpy(), "ECT Prediction", xlabel="Row (y)", ylabel="Depth (z)", font_size=24, colorbar=False, ticks=False, scale_bar=True, save_path=os.path.join(pred_path, f"pred_scale_bar_{i}_{j}.png"))

            draw_grid(perm[j][0].cpu().detach().numpy(), "Confocal Microscopy", xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)",  font_size=18, colorbar=False, save_path=os.path.join(truth_path, f"truth_{i}_{j}.png"))
            draw_grid(perm[j][0].cpu().detach().numpy(), "Confocal Microscopy", xlabel="Row (y)", ylabel="Depth (z)",  font_size=24, colorbar=False, ticks=False, scale_bar=True, save_path=os.path.join(truth_path, f"truth_scale_bar_{i}_{j}.png"))

    run_time = run_time / len(test_loader)
        
    return  predictions, ground_truth, run_time



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to training configuration.", required=True)
    parser.add_argument('--checkpoint', type=str, help="Path to pretrained model.", required=False)
    parser.add_argument('--best_model', type=str, help="Path to best model [For Testing].", required=False)
    parser.add_argument('--output_dir', type=str, help="Path to output directory.", default="logs/baseline")  

    args = parser.parse_args()

    config_path = args.config
    checkpoint = args.checkpoint 
    best_model = args.best_model
    output_dir = args.output_dir 
    
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
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True) 

    # Load pretrained model state 
    start_epoch = 0
    if checkpoint:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint)

    # Prepare model
    model = TUNet(embed_dim=200, num_measurements=num_measurements, max_measurement=2, output_dim=(100, 200))
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Parameters: ", num_params, flush=True)
    
    # summary(model, input_size=(1, num_measurements))
    # model.apply(weights_init)
    
    min_valid_loss = 1_000_000
    train_loss = []
    val_loss = []
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training Loop
    for i in range(start_epoch, start_epoch+epochs):

        train_avg_loss, val_avg_loss = train_one_epoch(model, train_loader, val_loader, optimizer, i, device)

        writer.add_scalar("Loss/train", train_avg_loss, i)
        writer.add_scalar("Loss/val", val_avg_loss, i)

        if min_valid_loss > val_avg_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f} ---> {val_avg_loss:.6f}) \t Saving The Model', flush=True)
           
            # Saving State Dict
            checkpoint = {'epoch': i + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            
            # save_ckp is very slow
            # save_ckp(checkpoint, is_best=True, checkpoint_dir=output_dir, best_model_path=os.path.join(output_dir, "best_model.pth"))
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

            min_valid_loss = val_avg_loss

        train_loss.append(train_avg_loss)
        val_loss.append(val_avg_loss)
        
        draw_curve(i, train_loss, val_loss, "MSE", output_dir)

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
    print(metrics, flush=True)

    stats, table = tabulate_runs([metrics], run_time, os.path.join(output_dir, "stats.json"))
    print(table.draw(), flush=True)
    
    writer.add_scalar("SSIM_acc", metrics["SSIM"])
    writer.add_scalar("MSE_acc", metrics["MSE"])
    writer.add_scalar("MAE_acc", metrics["MAE"])
    writer.add_scalar("PSNR_acc", metrics["PSNR"])
    writer.add_scalar("IoU_acc", metrics["IoU"])
    writer.add_scalar("CC_acc", metrics["CC"])

    writer.flush()
    writer.close()
    


if __name__== "__main__":
    main()