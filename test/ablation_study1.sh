#!/bin/bash
python train.py --name hifitts_vctk_pha2_G7L3_4x --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/mdct_hifitts_pha2_G7L3 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0 --abs_spectro --center --explicit_encoding --n_blocks_global 7 --n_blocks_local 3 --lr_sampling_rate 12000 --save_epoch_freq 30 \
&& \
python train.py --name hifitts_vctk_pha2_G5L3_4x --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/mdct_hifitts_pha2_G5L3 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0 --abs_spectro --center --explicit_encoding --n_blocks_global 5 --n_blocks_local 3  --lr_sampling_rate 12000 --save_epoch_freq 30 \
&& \
python train.py --name hifitts_vctk_pha2_G3L2_4x --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/mdct_hifitts_pha2_G3L2 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2  --lr_sampling_rate 12000 --save_epoch_freq 30 \
&& \
python train.py --name hifitts_vctk_pha2_G3L2_48ngf_4x --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/mdct_hifitts_pha2_G3L2_48ngf --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --lr_sampling_rate 12000 --save_epoch_freq 30