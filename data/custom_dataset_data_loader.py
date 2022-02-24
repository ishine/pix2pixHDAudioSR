import torch.utils.data
from data.base_data_loader import BaseDataLoader
from torch.utils.data import SubsetRandomSampler
import random
import os

def CreateDataset(opt):
    dataset = None
    if opt.phase == 'train':
        from data.audio_mdct_spectrogram_dataset import AudioMDCTSpectrogramDataset
        dataset = AudioMDCTSpectrogramDataset(opt)
    elif opt.phase == 'test':
        from data.audio_mdct_spectrogram_dataset import AudioMDCTSpectrogramTestDataset
        dataset = AudioMDCTSpectrogramTestDataset(opt)

    print("dataset [%s] was created" % (dataset.name()))
    #dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(torch.floor(torch.Tensor([opt.validation_split * dataset_size])))

        if opt.val_indices is not None:
            self.val_indices = torch.load(opt.val_indices)
            self.train_indices = torch.tensor(list(set(indices) - set(self.val_indices)))
        else:
            if not opt.serial_batches:
                random.seed(opt.seed)
                random.shuffle(indices)
            self.train_indices, self.val_indices = indices[split:], indices[:split]
            self.data_lenth = min(len(self.train_indices), self.opt.max_dataset_size)
            torch.save(self.val_indices, os.path.join(self.opt.checkpoints_dir, self.opt.name,'validation_indices.pt'))

        # Creating PT data samplers and loaders:
        if opt.phase == "train":
            train_sampler = SubsetRandomSampler(self.train_indices)
            valid_sampler = SubsetRandomSampler(self.val_indices)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                sampler=train_sampler,
                num_workers=int(opt.nThreads),
                pin_memory=True)
            if len(self.val_indices) != 0:
                self.eval_data_lenth = len(self.val_indices)
                self.eval_dataloder = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=opt.batchSize,
                    sampler=valid_sampler,
                    num_workers=int(opt.nThreads),
                    pin_memory=True)
            else:
                self.eval_dataloder = None
                self.eval_data_lenth = 0
        elif opt.phase == "test":
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                num_workers=int(opt.nThreads),
                shuffle=False,
                pin_memory=True)
            self.eval_dataloder = None
            self.eval_data_lenth = 0

    def load_data(self):
        return self.dataloader

    def eval_data(self):
        return self.eval_dataloder

    def eval_data_len(self):
        return self.eval_data_lenth

    def __len__(self):
        return self.data_lenth
