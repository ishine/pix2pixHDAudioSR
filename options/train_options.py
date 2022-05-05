from .base_options import BaseOptions
from .audio_config import *

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--eval_freq', type=int, default=2000, help='frequency of evaluating matrics')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--abs_spectro', action='store_true', help='use absolute value of spectrogram')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--validation_split', type=float, default=0.05, help='path to file containing validation split indices')
        self.parser.add_argument('--val_indices', type=str, help='proportion of training data to be used as validation data if validation_split is not specified')
        self.parser.add_argument('--eval_size', type=int, default=100, help='how many samples to evaluate')
        self.parser.add_argument('--phase_encoding_mode', type=str, default=None, help='norm_dist|uni_dist|None')

        # for discriminators
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_mat', type=float, default=50.0, help='weight for phase matching loss')
        self.parser.add_argument('--lambda_time', type=float, default=1.0, help='weight for time domain loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--use_match_loss', action='store_true', help='if specified, use matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--use_hifigan_D', action='store_true', help='if specified, use multi-scale-multi-period hifigan time domain discriminator')

        # STFT params
        self.parser.add_argument('--lr_sampling_rate', type=int, default=LR_SAMPLE_RATE, help='low resolution sampling rate')
        self.parser.add_argument('--hr_sampling_rate', type=int, default=HR_SAMPLE_RATE, help='high resolution sampling rate')
        self.parser.add_argument('--segment_length', type=int, default=FRAME_LENGTH, help='audio segment length')
        self.parser.add_argument('--n_fft', type=int, default=N_FFT, help='num of FFT points')
        self.parser.add_argument('--hop_length', type=int, default=HOP_LENGTH, help='sliding window increament')
        self.parser.add_argument('--win_length', type=int, default=WIN_LENGTH, help='sliding window width')
        self.parser.add_argument('--center', action='store_true', help='centered FFT')
        self.parser.add_argument('--is_lr_input', action='store_true', help='if specified, the audio generator will assert the input as low res. And it will only do upsampling.')
        self.isTrain = True
