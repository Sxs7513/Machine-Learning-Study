#!/usr/bin/env bash
python main.py ^
    --output_dir ./result/ ^
    --summary_dir ./result/log/ ^
    --mode inference ^
    --is_training False ^
    --task SRGAN ^
    --input_dir_LR ./test_images/ ^
    --num_resblock 16 ^
    --perceptual_mode VGG54 ^
    --pre_trained_model True ^
    --checkpoint ../train_data/pre_train_model/SRGAN_pre-trained/model-200000 