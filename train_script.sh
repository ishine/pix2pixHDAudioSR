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