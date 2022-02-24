import csv
import os
from numpy import ceil, concatenate
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as aF
from data.base_dataset import BaseDataset

class AudioMDCTSpectrogramDataset(BaseDataset):
    def __init__(self, opt) -> None:
        BaseDataset.__init__(self)
        self.lr_sampling_rate = opt.lr_sampling_rate
        self.hr_sampling_rate = opt.hr_sampling_rate
        self.segment_length = opt.segment_length
        self.n_fft = opt.n_fft
        self.hop_length = opt.hop_length
        self.win_length = opt.win_length
        self.audio_file = self.get_files(opt.dataroot)
        self.center = opt.center

        torch.manual_seed(opt.seed)

    def __len__(self):
        return len(self.audio_file)

    def name(self):
        return 'AudioMDCTSpectrogramDataset'

    def readaudio(self, file_path):
        metadata = torchaudio.info(file_path)
        max_audio_start = metadata.num_frames - self.segment_length
        if max_audio_start > 0:
            offset = torch.randint(low=0, high=max_audio_start, size=(1,)).item()
            waveform, orig_sample_rate = torchaudio.load(file_path, frame_offset=offset, num_frames=self.segment_length)
        else:
            print("Warning: %s is shorter than segment_length"%file_path, metadata.num_frames)
            waveform, orig_sample_rate = torchaudio.load(file_path)
        return waveform, orig_sample_rate

    def __getitem__(self, idx):
        file_path = self.audio_file[idx]
        try:
            waveform, orig_sample_rate = self.readaudio(file_path)
        except: #try next until success
            i = 1
            while 1:
                print('Load failed!')
                file_path = self.audio_file[idx+i]
                try:
                    waveform, orig_sample_rate = self.readaudio(file_path)
                    break
                except:
                    i += 1

        hr_waveform = aF.resample(waveform=waveform, orig_freq=orig_sample_rate, new_freq=self.hr_sampling_rate)
        lr_waveform = aF.resample(waveform=waveform, orig_freq=orig_sample_rate, new_freq=self.lr_sampling_rate)
        lr_waveform = aF.resample(waveform=lr_waveform, orig_freq=self.lr_sampling_rate, new_freq=self.hr_sampling_rate)
        #lr_waveform = aF.lowpass_biquad(waveform, sample_rate=self.hr_sampling_rate, cutoff_freq = self.lr_sampling_rate//2) #Meet the Nyquest sampling theorem
        hr = self.seg_pad_audio(hr_waveform)
        lr = self.seg_pad_audio(lr_waveform)
        return {'image': hr.squeeze(0), 'label': lr.squeeze(0), 'inst':0, 'feat':0, 'path': file_path}

    def get_files(self, file_path):
        if os.path.isdir(file_path):
            print("Searching for audio file")
            file_list = []
            for root, dirs, files in os.walk(file_path, topdown=False):
                for name in files:
                    if os.path.splitext(name)[1] == ".wav" or ".mp3" or ".flac":
                        file_list.append(os.path.join(root, name))
        else:
            print("Using csv file list")
            root, csv_file = os.path.split(file_path)
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                file_list = [os.path.join(root, item) for sublist in list(csv_reader) for item in sublist]
        print(len(file_list))
        return file_list

    def seg_pad_audio(self, waveform):
        if waveform.size(1) >= self.segment_length:
            waveform = waveform[0][:self.segment_length]
        else:
            waveform = F.pad(
                waveform, (0, self.segment_length - waveform.size(1)), 'constant'
            ).data
        return waveform
class AudioMDCTSpectrogramTestDataset(BaseDataset):
    def __init__(self, opt) -> None:
        BaseDataset.__init__(self)
        self.lr_sampling_rate = opt.lr_sampling_rate
        self.hr_sampling_rate = opt.hr_sampling_rate
        self.segment_length = opt.segment_length
        self.n_fft = opt.n_fft
        self.hop_length = opt.hop_length
        self.win_length = opt.win_length
        self.center = opt.center
        self.dataroot = opt.dataroot
        try:
            self.raw_audio, self.in_sampling_rate = torchaudio.load(self.dataroot)
            self.audio_len = self.raw_audio.size(-1)
            print("Audio length:", self.audio_len)
        except:
            self.raw_audio = []
            print("load audio failed")
            exit(0)
        if not opt.is_lr_input:
            self.lr_audio = aF.resample(waveform=self.raw_audio, orig_freq=self.in_sampling_rate, new_freq=self.lr_sampling_rate)
        self.lr_audio = aF.resample(waveform=self.lr_audio, orig_freq=self.lr_sampling_rate, new_freq=self.hr_sampling_rate)
        self.seg_audio = self.seg_pad_audio(self.lr_audio)

    def __len__(self):
        return self.seg_audio.size(0)

    def name(self):
        return 'AudioMDCTSpectrogramTestDataset'

    def __getitem__(self, idx):
        return {'image': torch.empty(1), 'label': self.seg_audio[idx,:].squeeze(0), 'inst':torch.empty(1), 'feat':torch.empty(1), 'path': self.dataroot}

    def seg_pad_audio(self, audio):
        audio = audio.squeeze(0)
        length = len(audio)
        if length >= self.segment_length:
            num_segments = int(ceil(length/self.segment_length))
            audio = F.pad(audio, (0, self.segment_length*num_segments - length), "constant").data
            audio = audio.unfold(dimension=0,size=self.segment_length,step=self.segment_length)
        else:
            audio = F.pad(audio, (0, self.segment_length - length), 'constant').data
            audio = audio.unsqueeze(0)

        return audio