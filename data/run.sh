 
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

## Bead datasets

# 04142022
python3.7 -m data.minerva --ect data/real/Larkin_Lab_Data/ECT_Train_04142022/ECT_1D_dataset/ECT_scan_1D_04142022_set_1.h5 --output_dir logs/0414022-set1-new --ect_cfg data/real/04142022_ECT.yaml
python3.7 -m data.confocal_ect --ect data/real/Larkin_Lab_Data/ECT_Train_04152022/ECT_1D_dataset/ECT_scan_1D_04152022_set_0.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_04152022/Confocal/Region_4_stat_Merged_RAW_ch00.tif --ect_cfg data/real/04152022_ECT.yaml --confocal_cfg data/real/04152022_Confocal.yaml --output_dir logs/04152022-set0

# 04152022
python3.7 -m data.minerva --ect data/real/Larkin_Lab_Data/ECT_Train_04152022/ECT_1D_dataset/ECT_scan_1D_04152022_set_1.h5 --output_dir logs/0415022-set1-new --ect_cfg data/real/04152022_ECT.yaml

# 04112022
python3.7 -m data.minerva --ect data/real/Rosenstein_Lab_Data/F0008_04112022_confocal_ECT_beads/ECT_1D_dataset/ECT_scan_04102022_set_0.h5 --output_dir logs/0411022-bead-dataset --ect_cfg data/real/04112022_ECT.yaml
python3.7 -m data.confocal_ect --ect data/real/Rosenstein_Lab_Data/F0008_04112022_confocal_ECT_beads/ECT_1D_dataset/ECT_scan_04102022_set_0.h5 --confocal data/real/Rosenstein_Lab_Data/F0008_04112022_confocal_ECT_beads//Confocal/Project_Full_CMOS_Scan_25percent_overlap_Merged_ch00-002.tif --ect_cfg data/real/04112022_ECT.yaml --confocal_cfg data/real/04112022_Confocal.yaml --output_dir logs/04112022-conf-ect
python3.7 -m data.confocal --confocal data/real/Rosenstein_Lab_Data/F0008_04112022_confocal_ECT_beads//Confocal/Project_Full_CMOS_Scan_25percent_overlap_Merged_ch00-002.tif  --confocal_cfg data/real/04112022_Confocal.yaml --output_dir data/logs/04112022_confocal

# 04102022
python3.7 -m data.confocal --confocal data/real/Larkin_Lab_Data/04102022/04102022/Confocal/Full_CMOS_scan_15percent_overlap_Merged_RAW_ch00.tif  --confocal_cfg data/real/04102022_Confocal.yaml --output_dir data/logs/04102022_confocal

# Biofilm Datasets
#07082022
python3.7 -m data.minerva --ect data/real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5 --ect_cfg data/real/07082022_ECT.yaml --output_dir logs/07082022_ect
python3.7 -m data.confocal_ect --ect data/real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif --ect_cfg data/real/07082022_ECT.yaml --confocal_cfg data/real/07082022_Confocal.yaml --output_dir logs/07082022_conf_ect
python3.7 -m data.conf_ect_sweep  --ect data/real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif --ect_cfg data/real/07082022_ECT.yaml --confocal_cfg data/real/07082022_Confocal.yaml --stride 4 
python3.7 -m data.confocal --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif  --confocal_cfg data/real/07082022_Confocal.yaml --output_dir data/logs/07082022_confocal

# Resized z-stack, 1 pixel -> 1um
python3.7 -m data.confocal --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif  --confocal_cfg data/real/07082022_Confocal2.yaml --output_dir data/logs/07082022_confocal2
python3.7 -m data.conf_ect_sweep  --ect data/real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif --ect_cfg data/real/07082022_ECT.yaml --confocal_cfg data/real/07082022_Confocal2.yaml --stride 4 --output_dir logs/test_sweep


# 06132023
python3.7 -m data.confocal --confocal data/real/Larkin_Lab_Data/ECT_Train_06132023/F0421_06132023/Confocal/CFP_Biofilm.tif  --confocal_cfg data/real/06132023_Confocal.yaml --output_dir data/logs/06132023_confocal
python3.7 -m data.minerva --ect data/real/Larkin_Lab_Data/ECT_Train_06132023/F0421_06132023/F0421_minerva/F0421_ect_pre_p2.h5_15pair --ect_cfg data/real/06132023_ECT.yaml --output_dir logs/06132023_ect
python3.7 -m data.confocal_ect --ect data/real/Larkin_Lab_Data/ECT_Train_06132023/F0421_06132023/F0421_minerva/F0421_ect_pre_p2.h5_15pair --confocal data/real/Larkin_Lab_Data/ECT_Train_06132023/F0421_06132023/Confocal/CFP_Biofilm.tif  --ect_cfg data/real/06132023_ECT.yaml --confocal_cfg data/real/06132023_Confocal.yaml --output_dir logs/06132023_conf_ect
nohup python3.7 -m data.conf_ect_sweep  --ect data/real/Larkin_Lab_Data/ECT_Train_06132023/F0421_06132023/F0421_minerva/F0421_ect_pre_p2.h5_15pair --confocal data/real/Larkin_Lab_Data/ECT_Train_06132023/F0421_06132023/Confocal/CFP_Biofilm.tif  --ect_cfg data/real/06132023_ECT.yaml --confocal_cfg data/real/06132023_Confocal.yaml --stride 4 --output_dir logs/06132023_conf_ect_sweep </dev/null >/dev/null 2>&1 &  

# 07112023
python3.7 -m data.confocal --confocal data/real/Larkin_Lab_Data/F0433_07112023/CFP_Region_4_Merged_RAW_ch00.tif   --confocal_cfg data/real/07112023_Confocal.yaml --output_dir data/logs/07112023_confocal
python3.7 -m data.minerva --ect data/real/Larkin_Lab_Data/F0433_07112023/F0433_minerva/F0433_ect_pre_p1_15pair.h5 --ect_cfg data/real/07112023_ECT.yaml --output_dir data/logs/07112023_ect
python3.7 -m data.confocal_ect --ect data/real/Larkin_Lab_Data/F0433_07112023/F0433_minerva/F0433_ect_pre_p1_15pair.h5 --confocal data/real/Larkin_Lab_Data/F0433_07112023/CFP_Region_4_Merged_RAW_ch00.tif --ect_cfg data/real/07112023_ECT.yaml --confocal_cfg data/real/07112023_Confocal.yaml --output_dir logs/07112023_conf_ect
nohup python3.7 -m data.conf_ect_sweep  --ect data/real/Larkin_Lab_Data/F0433_07112023/F0433_minerva/F0433_ect_pre_p1_15pair.h5 --confocal data/real/Larkin_Lab_Data/F0433_07112023/CFP_Region_4_Merged_RAW_ch00.tif --ect_cfg data/real/07112023_ECT.yaml --confocal_cfg data/real/07112023_Confocal.yaml --stride 4   --output_dir data/logs/07112023_conf_ect_sweep  </dev/null >/dev/null 2>&1 &

# 08062023
python3.7 -m data.confocal --confocal data/real/Larkin_Lab_Data/ECT_Train_08062023/F0445_lif_Region_1_Merged.tif   --confocal_cfg data/real/08062023_Confocal.yaml --output_dir data/logs/08062023_confocal
python3.7 -m data.minerva --ect data/real/Larkin_Lab_Data/ECT_Train_08062023/F0445_minerva/F0445_ect_pre_p1_15pair.h5 --ect_cfg data/real/08062023_ECT.yaml --output_dir data/logs/08062023_ect
python3.7 -m data.confocal_ect --ect data/real/Larkin_Lab_Data/ECT_Train_08062023/F0445_minerva/F0445_ect_pre_p1_15pair.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_08062023/F0445_lif_Region_1_Merged.tif --ect_cfg data/real/08062023_ECT.yaml --confocal_cfg data/real/08062023_Confocal.yaml --output_dir logs/08062023_conf_ect
nohup python3.7 -m data.conf_ect_sweep  --ect data/real/Larkin_Lab_Data/ECT_Train_08062023/F0445_minerva/F0445_ect_pre_p1_15pair.h5 --confocal data/real/Larkin_Lab_Data/ECT_Train_08062023/F0445_lif_Region_1_Merged.tif --ect_cfg data/real/08062023_ECT.yaml --confocal_cfg data/real/08062023_Confocal.yaml --stride 4   --output_dir data/logs/08062023_conf_ect_sweep  </dev/null >/dev/null 2>&1 &

