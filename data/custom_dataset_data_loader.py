import torch.utils.data
from data.base_data_loader import BaseDataLoader


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
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
