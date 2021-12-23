import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from util.spectro_img import compute_visuals
from .base_model import BaseModel
from . import networks
from .mdct import MDCT
import torchaudio.functional as aF

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_pha_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, use_pha_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, g_pha, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,g_pha,d_real,d_fake),flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, opt.use_pha_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)


            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_pha','D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def mdct(self, audio, min_value=1e-7, mask=False, norm_param=None, mask_mode=None, explicit_encoding=False, phase_encoding_mode=None):
        _mdct = MDCT(torch.kaiser_window(self.opt.win_length).cuda(), step_length=self.opt.hop_length, n_fft=self.opt.n_fft, center=self.opt.center, device = 'cuda').cuda()
        audio = _mdct(audio.cuda()).unsqueeze(1)
        log_audio = aF.amplitude_to_DB(
            (audio.abs() + min_value),20,min_value,1
            ).cuda()
        pha = torch.sign(audio).cuda()

        if norm_param is None:
            mean = log_audio.mean()
            std  = log_audio.var().sqrt()
        else:
            mean = norm_param['mean']
            std  = norm_param['std']

        log_audio = (log_audio-mean)/std

        if norm_param is None:
            audio_max = log_audio.max()
            audio_min = log_audio.min()
        else:
            audio_max = norm_param['max']
            audio_min = norm_param['min']

        if explicit_encoding:
            # multiply phase with log magnitude
            log_audio = (log_audio-audio_min)/(audio_max-audio_min)
            # log_audio @ [0,1]
            log_audio = log_audio*pha
            # log_audio @ [-1,1], double peak
        else:   #TODO
            if   phase_encoding_mode == 'uni_dist':
                pha = pha*torch.rand(pha.size()).cuda()
            elif phase_encoding_mode == 'norm_dist':
                _noise = torch.randn(pha.size()).cuda()
                _noise_min = _noise.min()
                _noise_max = _noise.max()
                _noise = (_noise - _noise_min)/(_noise_max - _noise_min)
                pha = pha*_noise
            elif phase_encoding_mode == 'norm_dist2':
                _noise = torch.randn(pha.size()).abs().cuda()
                pha = pha*_noise
            elif phase_encoding_mode == 'scale':
                pha = pha*0.5
            log_audio = (log_audio-audio_min)/(audio_max-audio_min)
            # log_audio @ [-1,1], singal peak

        if mask:
            # mask the lr spectro so that it does not learn from garbage infomation
            size = log_audio.size()
            up_ratio = self.opt.hr_sampling_rate / self.opt.lr_sampling_rate
            mask_size = int(size[2]/up_ratio)
            _mask = torch.cat(
            (
                 torch.ones(size[0],
                            size[1],
                            mask_size,
                            size[3]
                            ),
                torch.zeros(size[0],
                            size[1],
                            size[2]-mask_size,
                            size[3]
                            )
            ),dim=2).cuda()
            log_audio = log_audio * _mask

            if mask_mode is not None:
                # fill the blank mask with noise
                _noise = torch.randn(size[0], size[1], size[2]-mask_size, size[3]).cuda()
                _noise_min = _noise.min()
                _noise_max = _noise.max()

                if mask_mode == 'mode0':
                    #fill empty with randn noise, single peak, centered at 0
                    _noise = _noise/(_noise_max - _noise_min)
                    #_noise @ [-1,1]
                elif mask_mode == 'mode1':
                    #fill empty with randn noise, double peak, mimic the real distribution
                    _noise = (_noise - _noise_min)/(_noise_max - _noise_min)
                    #_noise @ [0,1]
                    psudo_pha = 2*torch.randint(low=0,high=2,size=_noise.size()).cuda()-1
                    _noise = _noise * psudo_pha
                    #_noise @ [-1,1]
                elif mask_mode == 'mode2':
                    #fill empty with randn noise, single peak, centered at 0.5
                    _noise = (_noise - _noise_min)/(_noise_max - _noise_min)

                noise = torch.cat(
                        (
                            torch.zeros(size[0],
                                        size[1],
                                        mask_size,
                                        size[3]
                                        ).cuda(),
                            _noise
                        ),dim=2).cuda()
                log_audio = log_audio + noise

        return log_audio, pha, {'max':audio_max, 'min':audio_min, 'mean':mean, 'std':std}

    def encode_input(self, lr_audio, inst_map=None, hr_audio=None, feat_map=None, infer=False):
        # real images for training
        if hr_audio is not None:
            hr_spectro, hr_pha, norm_param = self.mdct(hr_audio.data.cuda(), mask = False, norm_param=None, min_value=self.opt.min_value, mask_mode=None, explicit_encoding=self.opt.explicit_encoding, phase_encoding_mode=self.opt.phase_encoding_mode)
            hr_spectro = Variable(hr_spectro.data.cuda())

        lr_spectro, lr_pha, _norm_param = self.mdct(lr_audio, mask = self.opt.mask, norm_param=None, min_value=self.opt.min_value, mask_mode=self.opt.mask_mode, explicit_encoding=self.opt.explicit_encoding, phase_encoding_mode=self.opt.phase_encoding_mode)
        lr_spectro = lr_spectro.data.cuda()

        """ if self.opt.label_nc == 0:
            lr_spectro = lr_spectro.data.cuda()
        else:
            # create one-hot vector for label map
            size = lr_spectro.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, lr_spectro.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half() """

        # (deprecated) get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            #edge_map = self.get_edges(inst_map)
            lr_spectro = torch.cat((lr_spectro, inst_map), dim=1)
        lr_spectro = Variable(lr_spectro, volatile=infer)

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            """ if self.opt.label_feat:
                #inst_map = label_map.cuda()
                inst_map = lr_pha.cuda() """

        return lr_spectro, lr_pha, hr_spectro, hr_pha, feat_map, inst_map, norm_param

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, lr_audio, inst, hr_audio, feat, infer=False):
        # Encode Inputs
        lr_spectro, lr_pha, hr_spectro, hr_pha, feat_map, inst_map, norm_param = self.encode_input(lr_audio, inst, hr_audio, feat)
        if not self.opt.explicit_encoding and self.opt.input_nc>=2:
            lr_spectro = torch.cat((lr_spectro, lr_pha), dim=1)
            hr_spectro = torch.cat((hr_spectro, hr_pha), dim=1)
        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                # for training
                # todo: implement multiple encoding method
                # use lr_pha temporaily. It will be replaced by inst_map for general propose
                feat_map = self.netE.forward(hr_spectro, lr_pha)
            # when inferrencing, it will select one from kmeans
            input_concat = torch.cat((lr_spectro, feat_map), dim=1)
        else:
            input_concat = lr_spectro
        sr_result = self.netG.forward(input_concat)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(lr_spectro, sr_result, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(lr_spectro, hr_spectro)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((lr_spectro, sr_result), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # (deprecated) VGG feature matching loss
        loss_G_VGG = 0
        """ if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(sr_result, hr_spectro) * self.opt.lambda_feat """

        # Phase matching loss
        # If all phase are generated correctly, loss_G_pha will be 0
        # If using STFT instead of MDCT, a cosine function can be used for evaluating the phase distance
        loss_G_pha = 0
        if self.opt.input_nc>=2:
            sr_pha = torch.sign(sr_result)[:,0,:,:].unsqueeze(1) if self.opt.explicit_encoding else sr_result[:,1,:,:].unsqueeze(1)
            if self.opt.use_pha_loss:
                loss_G_pha = (1-torch.cos(0.5*torch.pi*(sr_pha-hr_pha))).mean()

        # Register current samples
        self.current_lable     = lr_spectro.detach().cpu().numpy()[0,0,:,:]
        self.current_generated = sr_result.detach().cpu().numpy()[0,0,:,:]
        self.current_real      = hr_spectro.detach().cpu().numpy()[0,0,:,:]
        if self.opt.input_nc>=2:
            self.current_lable_pha     = torch.sign(lr_pha).detach().cpu().numpy()[0,0,:,:]
            self.current_generated_pha = torch.sign(sr_pha).detach().cpu().numpy()[0,0,:,:]
            self.current_real_pha      = torch.sign(hr_pha).detach().cpu().numpy()[0,0,:,:]
        else:
            self.current_lable_pha = None
            self.current_generated_pha = None
            self.current_real_pha = None

        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_pha, loss_D_real, loss_D_fake ), None if not infer else sr_result ]

    def inference(self, lr_audio, inst, hr_audio=None):
        # Encode Inputs
        hr_audio = Variable(hr_audio) if hr_audio is not None else None
        lr_spectro, lr_pha, hr_spectro, hr_pha, feat_map, inst_map, norm_param  = self.encode_input(Variable(lr_audio), Variable(inst), hr_audio, infer=True)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(hr_spectro, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((lr_spectro, feat_map), dim=1)
        else:
            input_concat = lr_spectro

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                hr_spectro = self.netG.forward(input_concat)
        else:
            hr_spectro = self.netG.forward(input_concat)
        return hr_spectro, lr_pha, norm_param

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_current_visuals(self):
        lable_sp, lable_hist, _ = compute_visuals(sp=self.current_lable, abs=self.opt.abs_spectro)
        _, _, lable_pha = compute_visuals(pha=self.current_lable_pha)
        generated_sp, generated_hist, _ = compute_visuals(sp=self.current_generated, abs=self.opt.abs_spectro)
        _, _, generated_pha = compute_visuals(pha=self.current_generated_pha)
        real_sp, real_hist, _ = compute_visuals(sp=self.current_real, abs=self.opt.abs_spectro)
        _, _, real_pha = compute_visuals(pha=self.current_real_pha)
        if self.opt.input_nc>=2:
            return {'lable_spectro':        lable_sp,
                    'generated_spectro':    generated_sp,
                    'real_spectro':         real_sp,
                    'lable_hist':           lable_hist,
                    'generated_hist':       generated_hist,
                    'real_hist':            real_hist,
                    'lable_pha':            lable_pha,
                    'generated_pha':        generated_pha,
                    'real_pha':             real_pha}
        else:
            return {'lable_spectro':        lable_sp,
                    'generated_spectro':    generated_sp,
                    'real_spectro':         real_sp,
                    'lable_hist':           lable_hist,
                    'generated_hist':       generated_hist,
                    'real_hist':            real_hist,}

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)