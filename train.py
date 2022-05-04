import math
import os
import time
import csv

import numpy as np
import torch
from torch.autograd import Variable

def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

from data.data_loader import CreateDataLoader
from models.mdct import IMDCT2
from models.models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util.util import compute_matrics

#import debugpy
#debugpy.listen(("localhost", 5678))
#debugpy.wait_for_client()

#os.environ['CUDA_VISIBLE_DEVICES']='0'
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
eval_dataset = data_loader.eval_data()
eval_dataset_size = data_loader.eval_data_len()
print('#training data = %d' % dataset_size)
print('#evaluating data = %d' % eval_dataset_size)

# Create the model
model = create_model(opt)
visualizer = Visualizer(opt)
optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

# IMDCT for evaluation
from util.util import kbdwin, imdct
from dct.dct import IDCT
_idct = IDCT()
_imdct = IMDCT2(window=kbdwin, win_length=opt.win_length, hop_length=opt.hop_length, n_fft=opt.n_fft, center=opt.center, out_length=opt.segment_length, device = 'cuda',idct_op=_idct)

if opt.fp16:
    from torch.cuda.amp import autocast as autocast
    from torch.cuda.amp import GradScaler
    # According to the offical tutorial, use only one GradScaler and backward losses separately
    # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
    scaler = GradScaler()


# Set frequency for displaying information and saving
opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
total_steps = (start_epoch-1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
eval_delta = total_steps % opt.eval_freq if opt.validation_split > 0 else -1

# Safe ctrl-c
end = False
import signal
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    global end
    end = True
signal.signal(signal.SIGINT, signal_handler)

# Evaluation process
# Wrap it as a function so that I dont have to free up memory manually
def eval_model():
    err = []
    snr = []
    snr_seg = []
    pesq = []
    lsd = []
    for j, eval_data in enumerate(eval_dataset):
        model.eval()
        lr_audio = eval_data['label']
        hr_audio = eval_data['image']
        with torch.no_grad():
            sr_spectro, lr_pha, norm_param, lr_spectro = model.inference(lr_audio, None)
            up_ratio = opt.hr_sampling_rate / opt.lr_sampling_rate
            sr_audio = imdct(spectro=sr_spectro, pha=lr_pha, norm_param=norm_param, _imdct=_imdct, up_ratio=up_ratio, explicit_encoding=opt.explicit_encoding)
            _mse,_snr_sr,_snr_lr,_ssnr_sr,_ssnr_lr,_pesq,_lsd = compute_matrics(hr_audio.squeeze(), lr_audio.squeeze(), sr_audio.squeeze(), opt)
            err.append(_mse)
            snr.append((_snr_lr, _snr_sr))
            snr_seg.append((_ssnr_lr, _ssnr_sr))
            pesq.append(_pesq)
            lsd.append(_lsd)
        if j >= opt.eval_size:
            break

    eval_result = {'err': np.mean(err), 'snr': np.mean(snr), 'snr_seg': np.mean(snr_seg), 'pesq': np.mean(pesq), 'lsd': np.mean(lsd)}
    with open(eval_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=eval_result.keys())
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(eval_result)
    print('Evaluation:', eval_result)
    model.train()

# Training...
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if end:
            print('exiting and saving the model at the epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
            exit(0)
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # Whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        if opt.fp16:
            with autocast():
                losses, generated = model(Variable(data['label']), Variable(data['inst']),Variable(data['image']), Variable(data['feat']), infer=save_fake)
        else:
            losses, generated = model(Variable(data['label']), Variable(data['inst']),Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # Sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.loss_names, losses))

        # Calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_mat',0) + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            #with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            # update the scaler only once per iteration
            #scaler.update()
        else:
            loss_G.backward()
            optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:
            #with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
        else:
            loss_D.backward()
            optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ### display output images
        if save_fake:
            visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if total_steps % opt.eval_freq == eval_delta:
            eval_model()
            torch.cuda.empty_cache()

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate()
