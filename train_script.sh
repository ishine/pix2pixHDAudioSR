python train.py --name mdct_explicit_phase_coding_mode0 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 10 --gpu_id 0 --nThreads 0 --explicit_encoding --mask --mask_mode mode0

python train.py --name mdct_explicit_phase_coding_mode1 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 10 --gpu_id 1 --nThreads 0 --explicit_encoding --mask --mask_mode mode1

python train.py --name mdct_implicit_phase_coding --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 2 --nThreads 0 --mask --instance_feat --feat_num 1

python train.py --name mdct_implicit_phase_coding_mask0 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 3 --nThreads 0 --mask --mask_mode mode0 --instance_feat --feat_num 1

python train.py --name mdct_implicit_phase_coding_mask0_normphasesoft --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 0 --nThreads 0 --mask --mask_mode mode0 --instance_feat --feat_num 1 --phase_encoding_mode norm_dist