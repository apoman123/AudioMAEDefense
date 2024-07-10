torchrun --standalone --nnodes 1 --nproc-per-node 2 pretraining.py --num_workers 10 --batch_size 16 --accum_steps 16 \
--model_type "waveform" --masking_mode "random" --masked_ratio 0.8 --save_path "/home/apoman123/data/nas07/PersonalData/apoman123/pretrain_wavemae_checkpoints" \
--log_dir "/home/apoman123/data/nas07/PersonalData/apoman123/pretrain_wavemae_checkpoints/logs"