# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

"""
    Implementation of https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8358975
    FNN Autoencoder 
"""
import os
import yaml
import time
import argparse
import torch
from torch import nn

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

from metrics.metrics import Metrics, tabulate_runs


class FNNAutoencoder(nn.Module):
    def __init__(self,  num_measurements=80, mesh_height=100, mesh_width=200):
        super(FNNAutoencoder, self).__init__()
        
        # encoder layers
        mesh_res = mesh_height*mesh_width

        self.encoder =  nn.Sequential(
            nn.Linear(mesh_res, 10000),
            nn.ReLU(),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, num_measurements),
            nn.ReLU()        
        )


        # decoder layers
        self.decoder =  nn.Sequential(
            nn.Linear(num_measurements, 5000),
            nn.ReLU(),
            nn.Linear(5000, 10000),
            nn.ReLU(),
            nn.Linear(10000, mesh_res),
            nn.ReLU(),
        )


    def forward(self, measurement, permittivity):
        x = nn.Flatten()(permittivity)
        # y^ = F(x)
        encoder_output = self.encoder(x)
       
        # x^ = G(y)
        decoder_output = self.decoder(measurement)
       
        # y~ = F(x^)
        encoder_output_2 = self.encoder(decoder_output)
       
        # x~ = G(y^)
        decoder_output_2 = self.decoder(encoder_output)

        return encoder_output, decoder_output, encoder_output_2, decoder_output_2
    

    def loss(self, measurement, permittivity, encoder_output, decoder_output, encoder_output_2, decoder_output_2):
        mse_loss = nn.MSELoss()
        alpha_1, alpha_2, alpha_3, alpha_4 = 1.0 / 8.0, 5.0 / 8.0 , 1.0 / 8.0 , 1.0 / 8.0

        x = nn.Flatten()(permittivity)
        y = measurement
        
        loss = alpha_1*mse_loss(encoder_output, y) + alpha_2*mse_loss(decoder_output, x) + \
              alpha_3*mse_loss(encoder_output_2, y)  + alpha_4*mse_loss(decoder_output_2, x)

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

        enc_out, dec_out, enc_out2, dec_out2 = model(input_g.float(), perm.float())
       
        loss = model.loss(vb.float(), perm.float(), enc_out, dec_out, enc_out2, dec_out2)

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
        enc_out, dec_out, enc_out2, dec_out2 = model(input_g.float(), perm.float())

        loss = model.loss(vb.float(), perm.float(), enc_out, dec_out, enc_out2, dec_out2)
        val_losses += loss.item() 


    train_avg_loss = train_losses / len(train_loader)
    val_avg_loss = val_losses / len(val_loader)

    print('Epoch: %0.2f | Training Loss: %.6f | Validation Loss: %0.6f'  % (epoch, train_avg_loss, val_avg_loss))

    return float(train_avg_loss), val_avg_loss 


def test(model, test_loader, config, output_dir, device):
    run_time = 0.0
    predictions = torch.tensor([])
    predictions = predictions.to(device)

    ground_truth = torch.tensor([])
    ground_truth = ground_truth.to(device)

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
        enc_out, dec_out, enc_out2, dec_out2 = model(input_g.float(), perm.float())
        run_time += time.time() - st 

        # smooth predictions
        predicted_perm = dec_out.reshape((-1, 1, perm.shape[2], perm.shape[3]))
        pred_perm_smoothed, _ = smooth_predictions(predicted_perm, perm, config.MODEL.HEAD_ACTIVATION, config.DATASET.POS_VALUE, config.DATASET.NEG_VALUE)
        pred_perm_smoothed = predicted_perm
        
        predictions = torch.cat((predictions, pred_perm_smoothed))
        ground_truth = torch.cat((ground_truth, perm))

        for j, pred in enumerate(pred_perm_smoothed): 
            pred_perm = pred
            draw_grid(pred_perm[0].cpu().detach().numpy(), "ECT Prediction", xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)", colorbar=False, font_size=18, save_path=os.path.join(pred_path, f"pred_{i}_{j}.png"))
            draw_grid(pred_perm[0].cpu().detach().numpy(), "ECT Prediction", xlabel="Row (y)", ylabel="Depth (z)", colorbar=False, scale_bar=True, ticks=False, font_size=24, save_path=os.path.join(pred_path, f"pred_scale_bar_{i}_{j}.png"))

            draw_grid(perm[j][0].cpu().detach().numpy(), "Ground Truth",  xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)", colorbar=False, save_path=os.path.join(truth_path, f"truth_{i}_{j}.png"))
    

    run_time = run_time / len(test_loader)
    
    return  predictions, ground_truth, run_time


def weights_init(m):
    classname = m.__class__.__name__
    if classname != "ReLU":
        nn.init.normal_(m.weight.data, 0.0, 0.02)
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to training configuration.", required=True)
    parser.add_argument('--checkpoint', type=str, help="Path to pretrained model.", required=False)
    parser.add_argument('--best_model', type=str, help="Path to best model path [Testing Mode].", required=False)
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

    # Prepare model 
    model = FNNAutoencoder(num_measurements=num_measurements, mesh_height=100, mesh_width=200)
    model = model.to(device)
    # summary(model, [(1, num_measurements), (1, 1, 100, 200)])
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Parameters: ", num_params)
    
    # model.apply(weights_init)

    min_valid_loss = 1_000_000
    train_loss = []
    val_loss = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    if checkpoint:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint)

    # Training Loop
    for i in range(start_epoch, start_epoch+epochs):
        train_avg_loss, val_avg_loss = train_one_epoch(model, train_loader, val_loader, optimizer, i, device)

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
        
        draw_curve(i, train_loss, val_loss, "MSE Loss", os.path.join(output_dir, ))

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
    model.load_state_dict(torch.load(best_model))
   
    predictions, ground_truth, run_time = test(model, test_loader, config, output_dir, device)

    metrics = Metrics(device=device)
    metrics = metrics.forward(predictions, ground_truth)
    print(metrics)

    save_path = os.path.join(output_dir, "metrics.json")
    stats, table = tabulate_runs([metrics], run_time, save_path)
    
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