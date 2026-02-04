import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from utils import *
from model import *
from dataset import *
import copy
import config
import csv

import warnings
warnings.filterwarnings("ignore")




class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=0.0001, beta_end=0.0200):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, y):
        with torch.no_grad():
            n = y.shape[0]
            # x = torch.randn((n, 1, 40, 48, 40)).to(config.device)
            ###### hjx ######
            x = torch.randn((n, 1, 28, 32, 28)).to(config.device)
            ###### hjx ######
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(config.device)
                predicted_noise = model(x, y, t)
                alpha = self.alpha[t][:, None, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None, None]
                beta = self.beta[t][:, None, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x



# First stage
def train_AAE():
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.999)) 
    disc = Discriminator().to(config.device)
    opt_disc = optim.Adam(disc.parameters(),lr=config.learning_rate,betas=(0.5, 0.999))

    average = 0

    for epoch in range(config.epochs):
        print("epoch:", epoch)
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","recon_loss","disc_loss_epoch"])
        
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task = config.train, stage = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        recon_loss_epoch=0
        disc_loss_epoch=0

        for idx, (Abeta, name) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            decoded_Abeta = model(Abeta)
            
            disc_real = disc(Abeta)
            disc_fake = disc(decoded_Abeta)

            # print(Abeta.shape)
            # print(decoded_Abeta.shape)

            recon_loss = torch.abs(Abeta - decoded_Abeta).mean()
            g_loss = -torch.mean(disc_fake)
            loss = recon_loss*config.Lambda + g_loss

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            disc_loss = (d_loss_real + d_loss_fake)/2

            opt_model.zero_grad()
            loss.backward(retain_graph=True)

            opt_disc.zero_grad()
            disc_loss.backward()

            opt_model.step()
            opt_disc.step()

            recon_loss_epoch = recon_loss_epoch + recon_loss
            disc_loss_epoch = disc_loss_epoch + disc_loss
        
        writer.writerow([epoch+1, recon_loss_epoch.item()/length, disc_loss_epoch.item()/length])
        lossfile.close()

        #validation part
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.validation, stage= "validation")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        psnr_0 = 0
        ssim_0 = 0

        csvfile = open("result/"+str(config.exp)+"validation.csv", 'a+',newline = '')
        writer = csv.writer(csvfile)
        if epoch == 0:
            writer.writerow(['Epoch','PSNR','SSIM'])

        for idx, (Abeta, name) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            
            decoded_Abeta = model(Abeta)
            decoded_Abeta = torch.clamp(decoded_Abeta,0,1)

            # decoded_Abeta = decoded_Abeta.detach().cpu().numpy()
            # decoded_Abeta = np.squeeze(decoded_Abeta)
            # decoded_Abeta = decoded_Abeta.astype(np.float32)

            # Abeta = Abeta.detach().cpu().numpy()
            # Abeta = np.squeeze(Abeta)
            # Abeta = Abeta.astype(np.float32)
        
            # psnr_0 += round(psnr(Abeta,decoded_Abeta),3)  # 保留3位小数
            # ssim_0 += round(ssim(Abeta,decoded_Abeta),3)

            ###### hjx ######
            # psnr_val = psnr(Abeta, decoded_Abeta, data_range=1.0)   # [0, 1] -》 data_range=1
            # ssim_val = ssim(Abeta, decoded_Abeta, data_range=1.0)
            psnr_val, ssim_val = batch_psnr_ssim_torchmetrics(Abeta, decoded_Abeta, data_range=1.0)
            psnr_0 += psnr_val
            ssim_0 += ssim_val
            ###### hjx ######
        
        average_epoch = psnr_0/length + ssim_0 * 10/length
        writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
        csvfile.close()
        
        #### test part ####
        # if average_epoch > average: 
        if (average_epoch > average) or (epoch+1)%100 == 0: ###### hjx ######
            if average_epoch > average:
                average = average_epoch
            save_checkpoint(model, opt_model, filename=f"{config.CHECKPOINT_AAE}_epoch{epoch+1}")   ###### hjx ######

            dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.test, stage= "test")
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)
            length = dataset.length_dataset
            psnr_0 = 0
            ssim_0 = 0
            
            csvfile = open("result/"+str(config.exp)+"test.csv", 'a+',newline = '')
            writer = csv.writer(csvfile)
            if epoch == 0:
                writer.writerow(['Epoch','PSNR','SSIM'])

            for idx, (Abeta, name) in enumerate(loop):
                Abeta = np.expand_dims(Abeta, axis=1)
                Abeta = torch.tensor(Abeta)
                Abeta = Abeta.to(config.device)
                
                decoded_Abeta = model(Abeta)
                decoded_Abeta = torch.clamp(decoded_Abeta,0,1)

                # decoded_Abeta = decoded_Abeta.detach().cpu().numpy()
                # decoded_Abeta = np.squeeze(decoded_Abeta)
                # decoded_Abeta = decoded_Abeta.astype(np.float32)

                # Abeta = Abeta.detach().cpu().numpy()
                # Abeta = np.squeeze(Abeta)
                # Abeta = Abeta.astype(np.float32)
            
                # psnr_0 += round(psnr(Abeta,decoded_Abeta),3)
                # ssim_0 += round(ssim(Abeta,decoded_Abeta),3)
                ###### hjx ######
                # psnr_test = psnr(Abeta, decoded_Abeta, data_range=1.0)
                # ssim_test = ssim(Abeta, decoded_Abeta, data_range=1.0)
                psnr_test, ssim_test = batch_psnr_ssim_torchmetrics(Abeta, decoded_Abeta, data_range=1.0)
                psnr_0 += psnr_test
                ssim_0 += ssim_test
                ###### hjx ######

            writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
            csvfile.close()

def encoding():
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    image = nib.load(config.path)

    dataset = OneDataset(root_Abeta=config.whole_Abeta, task = config.train, stage= "Non")
    loader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.numworker,pin_memory=True)
    loop = tqdm(loader, leave=True)

    for idx, (Abeta,name) in enumerate(loop):
        Abeta = np.expand_dims(Abeta, axis=1)
        Abeta = torch.tensor(Abeta)
        Abeta = Abeta.to(config.device)
        
        latent_Abeta = model.encoder(Abeta)
        latent_Abeta = latent_Abeta.detach().cpu().numpy()
        latent_Abeta = np.squeeze(latent_Abeta)
        latent_Abeta = latent_Abeta.astype(np.float32)

        latent_Abeta = nib.Nifti1Image(latent_Abeta, image.affine)  # 根据MRI的仿射矩阵进行存储
        nib.save(latent_Abeta, config.latent_Abeta+str(name[0]))



###### hjx ######
def testAAE():
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.test, stage= "test")
    loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
    loop = tqdm(loader, leave=True)
    length = dataset.length_dataset
    psnr_0 = 0
    ssim_0 = 0

    for idx, (Abeta, name) in enumerate(loop):
        Abeta = np.expand_dims(Abeta, axis=1)
        Abeta = torch.tensor(Abeta)
        Abeta = Abeta.to(config.device)
        
        decoded_Abeta = model(Abeta)
        decoded_Abeta = torch.clamp(decoded_Abeta,0,1)

        psnr_test, ssim_test = batch_psnr_ssim_torchmetrics(Abeta, decoded_Abeta, data_range=1.0)
        psnr_0 += psnr_test
        ssim_0 += ssim_test

    print(f"psnr: {psnr_0/length}")
    print(f"ssim: {ssim_0/length}")


def encoding_test():
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    image = nib.load(config.path)

    dataset = OneDataset(root_Abeta=config.whole_Abeta, task = config.test, stage= "test")
    loader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.numworker,pin_memory=True)
    loop = tqdm(loader, leave=True)

    for idx, (Abeta,name) in enumerate(loop):
        Abeta = np.expand_dims(Abeta, axis=1)
        Abeta = torch.tensor(Abeta)
        Abeta = Abeta.to(config.device)
        
        latent_Abeta = model.encoder(Abeta)
        latent_Abeta = latent_Abeta.detach().cpu().numpy()
        latent_Abeta = np.squeeze(latent_Abeta)
        latent_Abeta = latent_Abeta.astype(np.float32)

        latent_Abeta = nib.Nifti1Image(latent_Abeta, image.affine)  # 根据MRI的仿射矩阵进行存储
        nib.save(latent_Abeta, config.latent_Abeta+str(name[0]))
###### hjx ######





# Second stage
def train_LDM():
    gpus = config.gpus
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)

    Unet = UNet(image_size=config.image_size).to(config.device)     ### hjx
    opt_Unet= optim.AdamW(Unet.parameters(), lr=config.learning_rate)
    Unet = nn.DataParallel(Unet,device_ids=gpus,output_device=gpus[0])
    ema = EMA(0.9999)
    ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)

    L2 = nn.MSELoss()
    diffusion = Diffusion(noise_steps=config.noiseSteps)    ### hjx
    average = 0
    min_mse = 1.0 ###### hjx ######

    for epoch in range(config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","MSE_loss"])
        
        dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.latent_Abeta, task = config.train, stage = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset

        MSE_loss_epoch = 0

        for idx, (MRI, latent_Abeta, name, label) in enumerate(loop):
            label = label.to(config.device)
            MRI = np.expand_dims(MRI, axis=1)
            MRI = torch.tensor(MRI)
            MRI = MRI.to(config.device)
            latent_Abeta = np.expand_dims(latent_Abeta, axis=1)
            latent_Abeta = torch.tensor(latent_Abeta)
            latent_Abeta = latent_Abeta.to(config.device)

            t = diffusion.sample_timesteps(latent_Abeta.shape[0]).to(config.device)
            x_t, noise = diffusion.noise_images(latent_Abeta, t)
            predicted_noise = Unet(x_t, MRI, t, label)
            loss = L2(predicted_noise, noise)

            opt_Unet.zero_grad()
            loss.backward()
            opt_Unet.step()
            ema.step_ema(ema_Unet, Unet)

            MSE_loss_epoch += loss

        writer.writerow([epoch+1,MSE_loss_epoch.item()/length])
        lossfile.close()
        cur_mse = MSE_loss_epoch.item()/length  ###### hjx ######

        #### validation part ####
        dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.whole_Abeta, task = config.validation, stage = "validation")
        # ssim 计算需要 batch_size=1
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        psnr_0 = 0
        ssim_0 = 0

        csvfile = open("result/"+str(config.exp)+"validation.csv", 'a+',newline = '')
        writer = csv.writer(csvfile)
        if epoch == 0:
            writer.writerow(['Epoch','PSNR','SSIM'])

        
        if cur_mse < min_mse or (epoch+1)%100 == 0: ###### hjx ######
            if cur_mse < min_mse:    ###### hjx ######
                min_mse = cur_mse  
            for idx, (MRI, Abeta, name, label) in enumerate(loop):
                MRI = np.expand_dims(MRI, axis=1)
                MRI = torch.tensor(MRI)
                MRI = MRI.to(config.device)

                sampled_latent = diffusion.sample(ema_Unet, MRI)
                syn_Abeta = model.decoder(sampled_latent)
                syn_Abeta = torch.clamp(syn_Abeta,0,1)

                # syn_Abeta = syn_Abeta.detach().cpu().numpy()
                # syn_Abeta = np.squeeze(syn_Abeta)
                # syn_Abeta = syn_Abeta.astype(np.float32)

                # Abeta = Abeta.detach().cpu().numpy()
                # Abeta = np.squeeze(Abeta)
                # Abeta = Abeta.astype(np.float32)

                # psnr_0 += round(psnr(Abeta,syn_Abeta),3) 
                # ssim_0 += round(ssim(Abeta,syn_Abeta),3)

                ###### hjx ######
                # psnr_val = psnr(Abeta, syn_Abeta, data_range=1.0)
                # ssim_val = ssim(Abeta, syn_Abeta, data_range=1.0)
                Abeta = Abeta.to(config.device)
                psnr_val, ssim_val = batch_psnr_ssim_torchmetrics(Abeta, syn_Abeta, data_range=1.0)
                psnr_0 += psnr_val
                ssim_0 += ssim_val
                ###### hjx ######
            
            average_epoch = psnr_0/length + ssim_0 * 10/length
            writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
            csvfile.close()
        
            #### test part ####
            # if average_epoch > average:
            if (average_epoch > average) or (epoch+1)%100 == 0: ###### hjx ######
                if average_epoch > average:
                    average = average_epoch
                save_checkpoint(ema_Unet, opt_Unet, filename=f"{config.CHECKPOINT_Unet}_epoch{epoch+1}")   ###### hjx ######

                dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.whole_Abeta, task = config.test, stage = "test")
                loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
                loop = tqdm(loader, leave=True)
                length = dataset.length_dataset
                psnr_0 = 0
                ssim_0 = 0

                csvfile = open("result/"+str(config.exp)+"test.csv", 'a+',newline = '')
                writer = csv.writer(csvfile)
                if epoch == 0:
                    writer.writerow(['Epoch','PSNR','SSIM'])

                for idx, (MRI, Abeta, name, label) in enumerate(loop):
                    MRI = np.expand_dims(MRI, axis=1)
                    MRI = torch.tensor(MRI)
                    MRI = MRI.to(config.device)

                    sampled_latent = diffusion.sample(ema_Unet, MRI)
                    syn_Abeta = model.decoder(sampled_latent)
                    syn_Abeta = torch.clamp(syn_Abeta,0,1)

                    # syn_Abeta = syn_Abeta.detach().cpu().numpy()
                    # syn_Abeta = np.squeeze(syn_Abeta)
                    # syn_Abeta = syn_Abeta.astype(np.float32)

                    # Abeta = Abeta.detach().cpu().numpy()
                    # Abeta = np.squeeze(Abeta)
                    # Abeta = Abeta.astype(np.float32)
                
                    # psnr_0 += round(psnr(Abeta,syn_Abeta),3)
                    # ssim_0 += round(ssim(Abeta,syn_Abeta),3)
                    ###### hjx ######
                    # psnr_test = psnr(Abeta, syn_Abeta, data_range=1.0)
                    # ssim_test = ssim(Abeta, syn_Abeta, data_range=1.0)#, channel_axis=None)
                    Abeta = Abeta.to(config.device)
                    psnr_test, ssim_test = batch_psnr_ssim_torchmetrics(Abeta, syn_Abeta, data_range=1.0)
                    psnr_0 += psnr_test
                    ssim_0 += ssim_test
                    ###### hjx ######

                writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
                csvfile.close()





###### hjx ######
                
def batch_psnr_ssim(gt_np, pred_np, data_range=1.0):
    """
    gt_np, pred_np: numpy arrays with shape (B, D, H, W) or (B, 1, D, H, W)
    返回: (psnr_sum, ssim_sum) —— 已对 batch 求和，外面再除以样本总数即可
    """
    # 去掉通道维（若存在）
    if gt_np.ndim == 5 and gt_np.shape[1] == 1:
        gt_np   = gt_np[:, 0]
        pred_np = pred_np[:, 0]
    psnr_sum = 0.0
    ssim_sum = 0.0
    B = gt_np.shape[0]

    for b in range(B):
        psnr_sum += psnr(gt_np[b], pred_np[b], data_range=data_range)
        ssim_sum += ssim(gt_np[b], pred_np[b], data_range=data_range)
    return psnr_sum, ssim_sum

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

def batch_psnr_ssim_torchmetrics(gt: torch.Tensor, pred: torch.Tensor, data_range: float = 1.0):
    # 保证有通道维
    if gt.ndim == 4:
        gt = gt.unsqueeze(1)      # (B, 1, D, H, W)
    if pred.ndim == 4:
        pred = pred.unsqueeze(1)

    gt = gt.float()
    pred = pred.float()

    # dim = 除去 batch 维以外的所有维度
    dims = tuple(range(1, pred.ndim))   # 对 (C, D, H, W) 

    # -------- PSNR：按样本算 --------
    psnr_vals = peak_signal_noise_ratio(
        pred, gt,
        data_range=data_range,
        reduction='none',
        dim=dims
    ) 
    #print("psnr_vals shape:", psnr_vals.shape)

    # -------- SSIM：按样本算 --------
    ssim_vals = structural_similarity_index_measure(
        pred, gt,
        data_range=data_range,
        reduction='none',
        gaussian_kernel=False,
        kernel_size=7,
    )
    #print("ssim_vals shape:", ssim_vals.shape)

    # 按你原来的风格：batch 内求和，外面再除以总样本数
    psnr_sum = psnr_vals.sum().item()
    ssim_sum = ssim_vals.sum().item()
    return psnr_sum, ssim_sum


def generatation():
    
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    image = nib.load(config.path)

    gpus = config.gpus
    Unet = UNet(image_size=config.image_size).to(config.device) 
    opt_Unet= optim.AdamW(Unet.parameters(), lr=config.learning_rate)
    Unet = nn.DataParallel(Unet,device_ids=gpus,output_device=gpus[0])
    load_checkpoint(config.CHECKPOINT_Unet, Unet, opt_Unet, config.learning_rate)
    ema = EMA(0.9999)
    ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)

    L2 = nn.MSELoss()
    diffusion = Diffusion(noise_steps=config.noiseSteps)    ### hjx

    dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.latent_Abeta, task = config.test, stage = "test")
    loader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.numworker,pin_memory=True)
    loop = tqdm(loader, leave=True)

    for idx, (MRI, latent_Abeta, name, label) in enumerate(loop):
        MRI = np.expand_dims(MRI, axis=1)
        MRI = torch.tensor(MRI)
        MRI = MRI.to(config.device)

        sampled_latent = diffusion.sample(ema_Unet, MRI)
        syn_Abeta = model.decoder(sampled_latent)
        syn_Abeta = torch.clamp(syn_Abeta,0,1)
        syn_Abeta = syn_Abeta.detach().cpu().numpy()
        syn_Abeta = np.squeeze(syn_Abeta)
        syn_Abeta = syn_Abeta.astype(np.float32)

        syn_Abeta = nib.Nifti1Image(syn_Abeta, image.affine)  # 根据MRI的仿射矩阵进行存储
        nib.save(syn_Abeta, config.syn_Abeta+str(name[0]))


###### hjx ######



if __name__ == '__main__':
    seed_torch()

    import argparse

    parser = argparse.ArgumentParser(description="AAE and LDM Model Pipeline")

    # 添加可选参数，使用 action='store_true' 表示如果命令行出现了该参数，则其值为 True
    parser.add_argument('--train_aae', action='store_true', help='Train the AAE model')
    parser.add_argument('--test_aae', action='store_true', help='Test the AAE model')
    parser.add_argument('--enc', action='store_true', help='Run encoding')
    parser.add_argument('--enc_test', action='store_true', help='Run encoding test')
    parser.add_argument('--train_ldm', action='store_true', help='Train the LDM model')
    parser.add_argument('--gen', action='store_true', help='Generate images/data')

    args = parser.parse_args()

    # 根据参数逻辑调用函数
    if args.train_aae:
        print("Starting AAE training...")
        train_AAE()

    if args.test_aae:
        testAAE()   # hjx

    if args.enc:
        encoding()

    if args.enc_test:
        encoding_test() # hjx

    if args.train_ldm:
        print("Starting LDM training...")
        train_LDM()

    if args.gen:
        print("Starting LDM generatation...")
        generatation()  # hjx
