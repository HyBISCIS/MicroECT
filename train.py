# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  


import os
import argparse
import yaml
import torch 

from data.dataset import Dataset
from torch.utils.data import DataLoader

from model.model import Generator, ResidualGenerator, weights_init
from model.train import train_one_epoch, test, smooth_predictions
from model.visualize import draw_curve
from model.loss import get_loss 
from model.lr import get_scheduler
from metrics.metrics import Metrics, tabulate_runs

from config import combine_cfgs

from data.plot import draw_grid 
from experiments.tree_generator import TreeGenerator
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils import init_torch_seeds, save_ckp, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to training configuration.", required=False)
    parser.add_argument('--checkpoint', type=str, help="Path to pretrained model.", required=False)
    parser.add_argument('--exp_name', type=str, help="Experiment Name", required=False)

    args = parser.parse_args()

    config_path = args.config
    checkpoint = args.checkpoint 
    exp_name = args.exp_name 

    config = combine_cfgs(config_path)

    if not exp_name:
        exp_name = config.NAME 
    
    seed = config.SEED 
    batch_size = config.DATASET.BATCH_SIZE
    data_path = config.DATASET.PATH
    num_measurements = config.DATASET.NUM_MEASUREMENTS 
    normalize = config.DATASET.NORMALIZE 
    shuffle = config.DATASET.SHUFFLE
    standardize = config.DATASET.STANDARDIZE 
    smooth = config.DATASET.SMOOTH
    noise = config.DATASET.NOISE 
    noise_stdv = config.DATASET.NOISE_STDV
    pos_value = config.DATASET.POS_VALUE
    neg_value = config.DATASET.NEG_VALUE
    lr = config.SOLVER.LEARNING_RATE
    epochs = config.SOLVER.EPOCHS
    loss = config.SOLVER.LOSS 
    gamma = config.SOLVER.GAMMA
    alpha = config.SOLVER.ALPHA
    weights = config.SOLVER.WEIGHTS
    trainable_weights = config.SOLVER.TRAINABLE_WEIGHTS
    optimizer = config.SOLVER.OPTIMIZER
    lr_scheduler = config.SOLVER.LR_SCHEDULER
    lr_gamma = config.SOLVER.LR_GAMMA
    energy_factor = config.SOLVER.ENERGY_FACTOR
    ebm_weights = config.SOLVER.EBM_WEIGHTS
    train_split, _, _ = config.DATASET.TRAIN_VAL_TEST_SPLIT    
    model_type = config.MODEL.TYPE
    head_activation = config.MODEL.HEAD_ACTIVATION
    hidden_activation = config.MODEL.HIDDEN_ACTIVATION 
    

    init_torch_seeds(seed)

    save_path = os.path.join('experiments', exp_name)

    output_tree = TreeGenerator(root_dir=save_path)
    output_tree.generate()

    with open(os.path.join(save_path, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    writer = SummaryWriter(os.path.join(save_path, 'logs'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read dataset
    dataset = Dataset(data_path, shuffle=shuffle, normalize=normalize, standardize=standardize, smooth=smooth, pos_value=pos_value, neg_value=neg_value, device=device)

    train_length = int(len(dataset)*train_split)
    val_length = int((len(dataset) - train_length) / 2)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, val_length, val_length], generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True) 

    # Prepare model 
    if model_type == 'Vanilla-Decoder':
        model = Generator(input_dim=num_measurements, head_activation=head_activation, hidden_activation=hidden_activation)
    else: 
        model = ResidualGenerator(input_dim=num_measurements, head_activation=head_activation, hidden_activation=hidden_activation)
    
    model = model.to(device)
    summary(model, (num_measurements, 1, 1))

    model.apply(weights_init)

    # Prepare Solver and loss 
    loss_fn = get_loss(loss, gamma=gamma, alpha=alpha, weights=weights, trainable_weights=trainable_weights, energy_factor=energy_factor, ebm_weights=ebm_weights, device=device)

    # if trainable_weights: 
    #     params = [{'params': model.parameters()}, {'params': loss_fn.awl.parameters()}]
    # else: 
    params = [{'params': model.parameters()}]
        
    gen_opt = torch.optim.Adam(params, lr=lr, weight_decay=0.001)
    scheduler = get_scheduler(lr_scheduler, gen_opt, gamma=lr_gamma)

    min_valid_loss = 1_000_000
    train_loss = []
    val_loss = []

    # Load pretrained model state 
    start_epoch = 0
    if checkpoint:
        model, gen_opt, start_epoch = load_checkpoint(model, gen_opt, checkpoint)

    # Training Loop
    for i in range(start_epoch, start_epoch+epochs):
        train_avg_loss, val_avg_loss = train_one_epoch(model, gen_opt, loss_fn, train_loader, val_loader, i, device, noise=noise, noise_stdv=noise_stdv)
        scheduler.step()
        print("Learning rate: ", scheduler.get_lr())
        
        writer.add_scalar("Loss/train", train_avg_loss, i)
        writer.add_scalar("Loss/val", val_avg_loss, i)

        if min_valid_loss > val_avg_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f} ---> {val_avg_loss:.6f}) \t Saving The Model', flush=True)
           
            # Saving State Dict
            checkpoint = {'epoch': i + 1, 'state_dict': model.state_dict(),
                'optimizer': gen_opt.state_dict()}
            save_ckp(checkpoint, is_best=True, checkpoint_dir=output_tree.ckp_path, best_model_path=output_tree.best_model_path)
            # torch.save(model.state_dict(), output_tree.best_model_path)
           
            min_valid_loss = val_avg_loss

        train_loss.append(train_avg_loss)
        val_loss.append(val_avg_loss)
        
        draw_curve(i, train_loss, val_loss, loss, os.path.join(output_tree.root_dir, ))

        # save checkpoint every 50 epochs 
        if i % 50 == 0: 
            checkpoint = {
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': gen_opt.state_dict()
            }
            save_ckp(checkpoint, is_best=False, checkpoint_dir=output_tree.ckp_path, best_model_path=None)

    # Testing Loop
    model, _, _ = load_checkpoint(model, gen_opt, output_tree.best_model_path)
   
    loss, predictions, ground_truth = test(model, loss_fn, test_loader, config, output_tree, device)

    metrics = Metrics(device=device)
    metrics = metrics.forward(predictions, ground_truth)
    print(metrics, flush=True)

    stats, table = tabulate_runs([metrics], None, os.path.join(output_tree.root_dir, "metrics.json"))
    print(table.draw(), flush=True)

    print(loss_fn.awl.params, flush=True)
    
    writer.add_scalar("SSIM_acc", metrics["SSIM"])
    writer.add_scalar("MSE_acc", metrics["MSE"])
    writer.add_scalar("MAE_acc", metrics["MAE"])
    writer.add_scalar("PSNR_acc", metrics["PSNR"])

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
