export CUDA_VISIBLE_DEVICES=3
python robustness_evaluation.py --dataset "vctk" --batch_size 4 --model_type "waveform" --masking_mode "uniform" --masked_ratio 0.5 \
--defense_model_task "masked_nam" --noise_level 50 \ 
--defense_model_path "/data/nas07_smb/PersonalData/apoman123/finetune_masked_nam_wavemae_checkpoints/masked_nam_0.5_WaveMAE_epoch_50.pth" \
--classifier "/data/nas07_smb/PersonalData/apoman123/vctk_rawnet3/RawNet3_epoch_25.pth" \
--budget 50 --attack_type "pgd" --epsilon 8/255 --norm "inf"