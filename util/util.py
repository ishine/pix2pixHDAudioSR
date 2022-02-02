from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torchaudio.functional as aF
import pysepm

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def compute_matrics(hr_audio,lr_audio,sr_audio,opt):
    # Normalize sr_audio and hr_audio
    sr_audio = (sr_audio-sr_audio.min())/(sr_audio.max()-sr_audio.min())
    hr_audio = (hr_audio-hr_audio.min())/(hr_audio.max()-hr_audio.min())
    lr_audio = (lr_audio-lr_audio.min())/(lr_audio.max()-lr_audio.min())

    # Calculate error
    mse = ((sr_audio-hr_audio)**2).mean().item()

    # Calculate SNR
    snr_sr = 10*torch.log10(torch.mean(hr_audio**2)/torch.mean((sr_audio-hr_audio)**2)).item()
    snr_lr = 10*torch.log10(torch.mean(hr_audio**2)/torch.mean((lr_audio-hr_audio)**2)).item()

    # Calculate segmental SNR
    #ssnr_sr = pysepm.SNRseg(clean_speech=hr_audio.numpy(),  processed_speech=sr_audio.numpy(), fs=opt.hr_sampling_rate)
    #ssnr_lr = pysepm.SNRseg(clean_speech=hr_audio.numpy(),  processed_speech=lr_audio.numpy(), fs=opt.hr_sampling_rate)

    # Calculate PESQ
    """ if hr_audio.dim() > 1:
        hr_audio = hr_audio.squeeze()
        sr_audio = sr_audio.squeeze()
        for i in range(hr_audio.size(-2)):
            p = []
            h = hr_audio[i,:]
            s = sr_audio[i,:]
            try:
                pesq = pysepm.pesq(aF.resample(h, orig_freq=opt.hr_sampling_rate, new_freq=16000).numpy(), aF.resample(s, orig_freq=opt.hr_sampling_rate, new_freq=16000).numpy(), 16000)
                p.append(pesq)
            except:
                print('PESQ no utterance')

        pesq = np.mean(p)
    else:
        try:
            pesq = pysepm.pesq(aF.resample(hr_audio,orig_freq=opt.hr_sampling_rate, new_freq=16000).numpy(), aF.resample(sr_audio,orig_freq=opt.hr_sampling_rate, new_freq=16000).numpy(), 16000)
        except:
            pesq = 0 """

    # Calculte STFT loss(LSD)
    hr_stft = aF.spectrogram(hr_audio, n_fft=opt.n_fft, hop_length=opt.hop_length, win_length=opt.win_length, window=torch.kaiser_window(opt.win_length, device='cuda'), center=opt.center, pad=0, power=2, normalized=False)
    sr_stft = aF.spectrogram(sr_audio, n_fft=opt.n_fft, hop_length=opt.hop_length, win_length=opt.win_length, window=torch.kaiser_window(opt.win_length, device='cuda'), center=opt.center, pad=0, power=2, normalized=False)
    hr_stft_log = torch.log10(hr_stft+1e-6)
    sr_stft_log = torch.log10(sr_stft+1e-6)
    lsd = torch.sqrt(torch.mean((hr_stft_log-sr_stft_log)**2,dim=-2)).mean().item()

    return mse,snr_sr,snr_lr,0,0,0,lsd