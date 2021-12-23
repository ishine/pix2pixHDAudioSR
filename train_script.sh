python train.py --name mdct_explicit_phase_coding_mode0 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 10 --gpu_id 0 --nThreads 0 --explicit_encoding --mask --mask_mode mode0

python train.py --name mdct_explicit_phase_coding_mode1 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 10 --gpu_id 1 --nThreads 0 --explicit_encoding --mask --mask_mode mode1

python train.py --name mdct_implicit_phase_coding --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 2 --nThreads 0 --mask --instance_feat --feat_num 1

python train.py --name mdct_implicit_phase_coding_mask0 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 3 --nThreads 0 --mask --mask_mode mode0 --instance_feat --feat_num 1

python train.py --name mdct_nophase2 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 16 --gpu_id 1 --nThreads 0 --mask --mask_mode mode2

python generate_audio.py --name mdct_nophase2 --phase test --dataroot /root/VCTK-Corpus/wav48/p225/p225_003.wav --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 2 --serial_batches --nThreads 0 --mask --mask_mode mode2 --load_pretrain ./checkpoints/mdct_nophase2