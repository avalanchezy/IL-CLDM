import torch

# 优先用cuda:1，如果没有就用cuda:0，最后fallback到cpu
device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu"
noiseSteps = 1000   # ori:1000
image_size = 28 ## hjx:112/4=28
latent_dim = 1
learning_rate = 1e-4
batch_size = 2  ## ori:2
numworker = 2   ## ori:0
epochs = 1000   ## ori:1000
time_dim = 128
Lambda = 100
num_classes = 4 ## ori:2



whole_Abeta = "./data/process/whole_Abeta_112"
latent_Abeta = "./data/latent_Abeta/latent_Abeta_112/"
whole_MRI = "./data/process/whole_MRI_112"

exp = "exp_128_112_epoch1000_testjit3d_cfg1_step10/" ## hjx
syn_Abeta = "./data/syn_Abeta/syn_Abeta_testjit3d_cfg1_step10/" ## hjx

path = "./data/process/whole_MRI_112/002S2010.nii.gz"
gpus = [1]

CHECKPOINT_AAE = "result/"+exp+"AAE.pth.tar"
CHECKPOINT_Unet = "result/"+exp+"Unet.pth.tar"
CHECKPOINT_JiT = "result/"+exp+"JiT.pth.tar"

train = "data_info/train.txt"
validation = "data_info/val.txt"
test = "data_info/test.txt" ## hjx class4:test4

# ========== Neural ODE Settings ==========
ode_hidden_dim = 32       # Hidden channels in ODE function
ode_time_dim = 64         # Time embedding dimension
ode_num_blocks = 3        # Number of conv blocks in ODE function
ode_solver = "dopri5"     # ODE solver: 'dopri5', 'euler', 'rk4', 'midpoint'
ode_rtol = 1e-5           # Relative tolerance for adaptive solvers
ode_atol = 1e-7           # Absolute tolerance for adaptive solvers
ode_epochs = 500          # Epochs for ODE training (can be less than AAE)

# Timepoints (in months)
timepoints = [0, 6, 12, 18, 24]
baseline_time = 0
target_time = 24

# ODE checkpoint
CHECKPOINT_ODE = "result/" + exp + "ODE.pth.tar"