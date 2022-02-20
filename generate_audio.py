import torch
from models.mdct import IMDCT
import torchaudio
import torchaudio.functional as aF

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#audio segments = %d' % dataset_size)

model = create_model(opt)
spectro_mag = []
spectro_pha = []
norm_params = []
model.eval()
with torch.no_grad():
    for i, data in enumerate(dataset):
        hr_spectro, lr_pha, norm_param = model.module.inference(data['label'], None)
        print(hr_spectro.size())
        spectro_mag.append(hr_spectro.abs().squeeze(1))
        spectro_pha.append(lr_pha.squeeze(1))
        norm_params.append(norm_param)

def imdct(log_mag, pha, norm_param, min_value=1e-7):
    _imdct = IMDCT(torch.kaiser_window(opt.win_length).cuda(), step_length=opt.hop_length, n_fft=opt.n_fft, center=opt.center, out_length=opt.segment_length, device = 'cuda').cuda()
    log_mag = log_mag*(norm_param['max']-norm_param['min'])+norm_param['min']
    log_mag = log_mag*norm_param['std']+norm_param['mean']
    mag = aF.DB_to_amplitude(log_mag.cuda(),10,0.5)-min_value
    mag = mag*pha
    audio = _imdct(mag.cuda())
    return audio

audio = []
for m,p,n in zip(spectro_mag,spectro_pha,norm_params):
    audio.append(imdct(log_mag=m, pha=p, norm_param=n))

audio = torch.stack(audio).view(1,-1)
print(audio.size())
sr_path = "./save_sr.wav"
lr_path = "./save_lr.wav"
torchaudio.save(sr_path, audio, opt.hr_sampling_rate)
torchaudio.save(lr_path, data_loader.dataset.raw_audio, opt.hr_sampling_rate)