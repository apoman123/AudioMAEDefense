export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nnodes 1 --nproc-per-node 2 pretraining.py --num_workers 10 --batch_size 128 --accum_steps 2 \
--model_type "spectrogram" --masking_mode "random" --masked_ratio 0.8 --save_path "/home/apoman123/data/nas07/PersonalData/apoman123/pretrain_spectrogrammae_checkpoints" \
--log_dir "/home/apoman123/data/nas07/PersonalData/apoman123/pretrain_spectrogrammae_checkpoints/logs"  