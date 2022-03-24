import math
import os
import time
import csv

import numpy as np
import torch

def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

from data.data_loader import CreateDataLoader
from models.mdct import IMDCT2
from models.models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util.util import compute_matrics

os.environ['NCCL_P2P_DISABLE']='1'
# Get the training options
opt = TrainOptions().parse()
# Set the seed
torch.manual_seed(opt.seed)
# Set the path for save the trainning losses
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'eval.csv')

if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# Create the data loader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#eval data = %d' % dataset_size)
total_steps = (start_epoch-1) * dataset_size + epoch_iter

# Create the model
model = create_model(opt)
visualizer = Visualizer(opt)

# IMDCT for evaluation
from util.util import kbdwin, imdct
_imdct = IMDCT2(window=kbdwin, win_length=opt.win_length, hop_length=opt.hop_length, n_fft=opt.n_fft, center=opt.center, out_length=opt.segment_length, device = 'cuda')

# Safe ctrl-c
end = False
import signal
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    global end
    end = True
signal.signal(signal.SIGINT, signal_handler)

# Training...
err = []
snr = []
snr_seg = []
pesq = []
lsd = []
for epoch in range(start_epoch, opt.niter+1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if end:
            print('exiting at the epoch %d, iters %d' % (epoch, total_steps))
            exit(0)
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.eval()
        lr_audio = data['label']
        hr_audio = data['image']
        with torch.no_grad():
            sr_spectro, lr_pha, norm_param, lr_spectro = model.module.inference(lr_audio, None)
            up_ratio = opt.hr_sampling_rate / opt.lr_sampling_rate
            sr_audio = imdct(spectro=sr_spectro, pha=lr_pha, norm_param=norm_param, _imdct=_imdct, up_ratio=up_ratio, explicit_encoding=opt.explicit_encoding)
            _mse,_snr_sr,_snr_lr,_ssnr_sr,_ssnr_lr,_pesq,_lsd = compute_matrics(hr_audio.squeeze(), lr_audio.squeeze(), 2*sr_audio.squeeze(), opt)
            err.append(_mse)
            snr.append(_snr_sr)
            snr_seg.append(_ssnr_sr)
            pesq.append(_pesq)
            lsd.append(_lsd)

    eval_result = {'err': np.mean(err), 'snr': np.mean(snr), 'snr_seg': np.mean(snr_seg), 'pesq': np.mean(pesq), 'lsd': np.mean(lsd)}
    with open(eval_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=eval_result.keys())
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(eval_result)
    print('Evaluation:', eval_result)

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter, time.time() - epoch_start_time))