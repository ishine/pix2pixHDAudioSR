python train.py --name mdct_explicit_phase_coding_mode0 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 10 --gpu_id 0 --nThreads 0 --explicit_encoding --mask --mask_mode mode0

python train.py --name mdct_explicit_phase_coding_mode1 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 10 --gpu_id 1 --nThreads 0 --explicit_encoding --mask --mask_mode mode1

python train.py --name mdct_implicit_phase_coding --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 2 --nThreads 0 --mask --instance_feat --feat_num 1

python train.py --name mdct_implicit_phase_coding_mask0 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 3 --nThreads 0 --mask --mask_mode mode0 --instance_feat --feat_num 1

python train.py --name mdct_pretrain --dataroot /home/neoncloud/openslr/LibriSpeech/files.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 0 --nThreads 8 --mask --mask_mode mode2

python generate_audio.py --name mdct_nophase2 --phase test --dataroot /root/VCTK-Corpus/wav48/p225/p225_003.wav --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 2 --serial_batches --nThreads 0 --mask --mask_mode mode2 --load_pretrain ./checkpoints/mdct_nophase2

python train.py --name mdct_2048 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 0 --nThreads 8 --mask --mask_mode mode0 --n_fft 2048 --win_length 2048

python train.py --name mdct_hifitts_pretrain --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 30 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --segment_length 25500

python train.py --name mdct_VCTK_with_pretrain --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --segment_length 25500 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_local --continue_train

python train.py --name mdct_VCTK_with_pretrain_glob --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 30 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --segment_length 25500 --load_pretrain ./checkpoints/mdct_hifitts_pretrain --niter 50 --niter_decay 50

python train.py --name mdct_hifitts_pretrain_amp --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0.01 --abs_spectro --center

python generate_audio.py --name mdct_hifitts_pretrain_amp_gen --dataroot /root/VCTK-Corpus/wav48/p227/p227_004.wav --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 2 --serial_batches --nThreads 0 --mask --mask_mode mode2 --netG local --validation_split 0 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --gpu_id 2 --center --phase test --serial_batches

python train.py --name mdct_hifitts_pretrain_test --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0.01 --abs_spectro --center --eval_freq 400 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp

python train.py --name mdct_VCTK_with_pretrain_amp --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center

python train.py --name mdct_VCTK_with_pretrain_amp_2x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 24000

python train.py --name mdct_VCTK_with_pretrain_amp_3x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 16000

python train.py --name mdct_VCTK_with_pretrain_amp_4x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 12000

tar -zcvf mdct_VCTK_with_pretrain_amp_6x.tgz /root/pix2pixHD/checkpoints/mdct_VCTK_with_pretrain_amp && tar -zcvf mdct_VCTK_with_pretrain_amp_2x.tgz /root/pix2pixHD/checkpoints/mdct_VCTK_with_pretrain_amp_2x && tar -zcvf mdct_VCTK_with_pretrain_amp_3x.tgz /root/pix2pixHD/checkpoints/mdct_VCTK_with_pretrain_amp_3x && tar -zcvf mdct_VCTK_with_pretrain_amp_4x.tgz /root/pix2pixHD/checkpoints/mdct_VCTK_with_pretrain_amp_4x && tar -zcvf mdct_hifitts_pretrain_amp2.tgz /root/pix2pixHD/checkpoints/mdct_hifitts_pretrain_amp2

python train.py --name mdct_hifitts_pretrain_explict_pha2 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding
#G: 730,713,346 D: 5,531,522

python train.py --name mdct_VCTK_with_pretrain_explict_pha_6x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_explict_pha2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 8000 --explicit_encoding

python train.py --name mdct_VCTK_with_pretrain_explict_pha_4x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_explict_pha2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 12000 --explicit_encoding

python train.py --name mdct_VCTK_with_pretrain_explict_pha_3x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_explict_pha2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 16000 --explicit_encoding

python train.py --name mdct_VCTK_with_pretrain_explict_pha_2x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_explict_pha2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 24000 --explicit_encoding

## ablation study 
## 75,501,568 per n_blocks_global
python train.py --name mdct_hifitts_pha2_G7L3 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 7 --n_blocks_local 3
#G: 579710210  D: 5531522
python train.py --name mdct_hifitts_pha2_G5L3 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 5 --n_blocks_local 3
#G: 428707074 D: 5531522
python train.py --name mdct_hifitts_pha2_G3L2 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 3 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2
# G: 277408770 D: 5531522
# 295,168 per n_blocks_local

python train.py --name mdct_hifitts_pha2_G3L2_48ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 3 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48
# G: 156050690 D: 5531522

python train.py --name mdct_hifitts_pha2_G3L2_32ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 3 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 32
# G: 69363202 D: 5531522

python train.py --name mdct_hifitts_pha2_G3L2_24ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 24
# G: 39020930 D: 5531522

python train.py --name mdct_hifitts_pha2_G3L2_16ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 16
# G: 17346306 D: 5531522

python train.py --name mdct_hifitts_pha2_G3L2_8ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 8
# G: 4339330 D: 5531522

python generate_audio.py --name pha2_G3L2_48_2x_gen --dataroot /root/VCTK-Corpus/wav48/p227/p227_004.wav --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 4 --serial_batches --nThreads 0 --mask --mask_mode mode2 --netG local --validation_split 0 --load_pretrain ./checkpoints/hifitts_vctk_pha2_G3L2_48ngf_2x --gpu_id 2 --center --phase test --serial_batches --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --lr_sampling_rate 24000