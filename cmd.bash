# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  


## Training Commands

# 1. Train bead dataset
python3.7 train.py --config config/experiments/exp01_IoU.yaml --exp_name bead_wnoise > logs/bead_wnoise 2>&1 &

python3.7 train.py --config config/experiments/beads/bead_FL_L1_Dice_resgen.yaml --exp_name fl_l1_dice_resgen > experiments/beads/fl_l1_dice_resgen.log 2>&1 &

python3.7 train.py --config config/experiments/beads/bead_FL_L1_Dice_resgen.yaml --exp_name fl_l1_dice_resgen_trianable_weights > experiments/beads/fl_l1_dice_resgen_trainable_weights.log 2>&1 &

# with noise
python3.7 train.py --config config/experiments/beads/bead_FL_L1_Dice_resgen.yaml --exp_name fl_l1_dice_resgen_trianable_weights_no_bias > experiments/beads/fl_l1_dice_resgen_trainable_weights_no_bias.log 2>&1 &

python3.7 train.py --config config/experiments/beads/bead_FL_L1_Dice_resgen_noise.yaml --exp_name fl_l1_dice_resgen_noise > experiments/beads/fl_l1_dice_resgen_noise.log 2>&1 &

python3.7 train.py --config config/experiments/beads/bead_FL_L1_Dice_resgen_noise.yaml --exp_name fl_l1_dice_resgen_noise_final_0.02 > experiments/beads/fl_l1_dice_resgen_noise_final_0.02.log 2>&1 &

#2. Train biofilm dataset 
python3.7 train.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --exp_name biofilm_trainable_weight_2 > experiments/biofilm_trainable_weight_2.log 2>&1 &

python3.7 train.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --exp_name biofilm_new_dataset > experiments/biofilm_new_dataset.log 2>&1 &

python3.7 train.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --exp_name biofim_unified_focal_loss > experiments/biofim_unified_focal_loss.log 2>&1 &

# python3.7 train.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --exp_name biofilm_full_dataset > experiments/biofilm_full_dataset.log 2>&1 &

# 3. Taion on new biofilm dataset
nohup python3.7 train.py --config config/experiments/exp02_FL_L1_biofilm.yaml --exp_name biofilm_new_data > experiments/biofilm_new_data.log 2>&1 &

## Testing Commands 

# 1. Test on the experimental bead dataset
python3.7 evaluate.py --config config/experiments/exp01_bead_test.yaml --model experiments/exp-noise/best_model.pth --dataset_path logs/0411022-bead-dataset --output_dir logs/bead_test

python3.7 evaluate.py --config config/experiments/beads/bead_FL_L1_Dice_resgen.yaml --model experiments/beads/fl_l1_dice_resgen/best_model.pth  --output_dir logs/bead_test_2

python3.7 evaluate.py --config config/experiments/beads/bead_FL_L1_Dice_resgen.yaml --model experiments/beads/fl_l1_dice_resgen_trianable_weights/best_model.pth  --output_dir experiments/beads/fl_l1_dice_resgen_trianable_weights/synth_bead_eval > experiments/beads/fl_l1_dice_resgen_trianable_weights/synth_bead_eval.log 2>&1 & 

# 2. Test on the biofilm dataset 
python3.7 evaluate.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --split_dataset True --model experiments/biofilm_trainable_weight_2/best_model.pth --output_dir logs/biofilm_trainable_weight

python3.7 evaluate.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --split_dataset True --model experiments/biofilm_full_dataset/best_model.pth --output_dir logs/biofilm_full_dataset

######################################################################################################
##################################### Baselines ######################################################
######################################################################################################

###########
### FNN ###
###########

## Training ## 

#1. Train the FNN autoencoder on the bead dataset 
python3.7 -m baseline.fnn --config config/experiments/exp01_FNN.yaml --output experiments/baselines/fnn_bead > experiments/baselines/fnn_bead.log 2>&1 &

# Train the FNN autoencoder on the biofilm dataset 
python3.7 -m baseline.fnn --config config/experiments/exp01_FNN_biofilm.yaml --output experiments/baselines/fnn_bead > experiments/baselines//fnn_biofilm.log 2>&1 &

## Testing ## 
## Test FNN on synthetic bead dataset
python3.7 -m baseline.fnn --config config/experiments/baselines/fnn_synth_bead.yaml --best_model experiments/baselines/fnn_bead/best_model.pth --output experiments/baselines/fnn_bead/fnn_synth_bead_eval > experiments/baselines/fnn_bead/fnn_synth_bead_eval.log 2>&1 &

## Test FNN on experimental bead dataset
python3.7 -m baseline.fnn --config config/experiments/baselines/fnn_exp_bead.yaml --best_model experiments/baselines/fnn_bead/best_model.pth --output logs/baselines/fnn_exp_bead > logs/baselines/fnn_exp_bead.log 2>&1 &

###################
### FNN+CNN-AE ####
###################

#2. Train the FNN+CNN autoencoder on the bead dataset 
python3.7 -m baseline.pvpn --config config/experiments/exp01_pvpn.yaml --output experiments/baselines/pvpn_bead > experiments/baselines/pvpn_bead.log 2>&1 &

# Train the FNN+CNN  autoencoder on the biofilm dataset 
python3.7 -m baseline.pvpn --config config/experiments/exp01_pvpn_biofilm.yaml --output experiments/baselines/pvpn_biofilm > experiments/baselines/pvpn_biofilm.log 2>&1 &

## Test FNN+CNN autoencoder on synthetic bead dataset
python3.7 -m baseline.pvpn --config config/experiments/baselines/pvpn_synth_bead.yaml --best_model experiments/baselines/pvpn_bead/best_model.pth --output experiments/baselines/pvpn_bead/pvpn_synth_bead_eval > experiments/baselines/pvpn_bead/pvpn_synth_bead_eval.log  2>&1 &

## Test FNN+CNN autoencoder on experimental bead dataset
python3.7 -m baseline.pvpn --config config/experiments/baselines/pvpn_exp_bead.yaml --best_model experiments/baselines/pvpn_bead/best_model.pth --output logs/baselines/pvpn_exp_bead > logs/baselines/pvpn_exp_bead.log 2>&1 &

## Test FNN+CNN autoencoder on experimental biofilm dataset
python3.7 -m baseline.pvpn --config config/experiments/baselines/pvpn_exp_biofilm.yaml --best_model experiments/baselines/pvpn_biofilm/best_model.pth --output logs/baselines/pvpn_biofilm > logs/baselines/pvpn_biofilm.log 2>&1 &

###################
###### UNET #######
###################

# Train the UNET baseline on the bead dataset
python3.7 -m baseline.unet --config config/experiments/exp01_unet.yaml --output_dir experiments/baselines/unet_bead > experiments/baselines/unet_bead.logs 2>&1 &

# Train the UNET baseline on the biofilm dataset
python3.7 -m baseline.unet --config config/experiments/exp01_unet_biofilm.yaml --output_dir experiments/baselines/unet_biofilm > experiments/baselines/unet_biofilm.logs 2>&1 &

# Test UNET on the synthetic bead dataset 
python3.7 -m baseline.unet --config config/experiments/baselines/unet_synth_bead.yaml --best_model experiments/baselines/unet_bead/best_model.pth --output experiments/baselines/unet_bead/unet_synth_bead_eval > experiments/baselines/unet_bead/unet_synth_bead_eval.log 2>&1 &

# Test UNET on the experimental bead dataset 
python3.7 -m baseline.unet --config config/experiments/baselines/unet_exp_bead.yaml --best_model experiments/baselines/unet_bead/best_model.pth --output logs/baselines/unet_exp_bead > logs/baselines/unet_exp_bead.log 2>&1 &

# Test UNET on the experimental biofilm dataset 
python3.7 -m baseline.unet --config config/experiments/baselines/unet_biofilm_bead.yaml --best_model experiments/baselines/unet_biofilm/best_model.pth --output logs/baselines/unet_biofilm > logs/baselines/unet_biofilm.log 2>&1 &


###################
### Tikhonov ######
###################

#4. Run the Tikhonov
# Test Tikhonov on the synthetic bead dataset
python3.7 -m baseline.jac --config config/experiments/baselines/jac_synth_bead.yaml --output logs/baselines/jac_synth_bead > logs/baselines/jac_synth_bead.log 2>&1 &

# Test Tikhonov on the experimental bead dataset
python3.7 -m baseline.jac --config config/experiments/baselines/jac_exp_bead.yaml --output logs/baselines/jac_exp_bead > logs/baselines/jac_exp_bead.log 2>&1 &

# Test Tikhonov on the biofilm test dataset 
python3.7 -m baseline.jac --config config/experiments/baselines/jac_biofilm.yaml --max_len 4 --output logs/baselines/jac_biofilm > logs/baselines/jac_bioiflm.log 2>&1 &

python -m baseline.inverse_biofilm --ect data/real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif --ect_cfg data/real/07082022_ECT.yaml --confocal_cfg data/real/07082022_Confocal.yaml --output_dir logs/baselines/jac_biofilm > logs/baselines/jac_biofilm.log 2>&1 & 

###################
###### GREIT ######
###################

#5. Run GREIT 
# Test GREIT on the experimental bead dataset
python3.7 -m baseline.greit --config config/experiments/baselines/greit_exp_bead.yaml --output logs/baselines/greit_exp_bead > logs/baselines/greit_exp_bead.logs 2>&1 &

# Test GREIT on the synthetic bead dataset
python3.7 -m baseline.greit --config config/experiments/baselines/greit_synth_bead.yaml --output logs/baselines/greit_synth_bead > logs/baselines/greit_synth_bead.logs 2>&1 &


##########################
### Column Predictions ###
##########################
python3.7 evaluate_column.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --model experiments/biofilm_lr_gamma_97/best_model.pth --ect data/real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif --ect_cfg data/real/07082022_ECT.yaml --confocal_cfg data/real/07082022_Confocal2.yaml --slice_col 150 --output_dir logs/cols_150 > logs/eval.log 2>&1

python3.7 evaluate_column.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --model experiments/biofilm_unified_focal_loss/best_model_1.pth --ect data/real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif --ect_cfg data/real/07082022_ECT.yaml --confocal_cfg data/real/07082022_Confocal2.yaml --slice_col 150 --output_dir logs/new_dataset_column

python3.7 evaluate_column_shifted.py --config config/experiments/exp01_FL_L1_Dice_biofilm.yaml --model experiments/biofilm_lr_gamma_97/best_model.pth --ect data/real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif --ect_cfg data/real/07082022_ECT.yaml --confocal_cfg data/real/07082022_Confocal2.yaml --slice_col 211 --output_dir logs/new_dataset_column