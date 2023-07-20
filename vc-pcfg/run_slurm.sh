#!/bin/bash
#SBATCH --job-name=babylm
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32Gb
#SBATCH --partition=main

module load anaconda/3
conda activate babylm-env
#python ./train_lm.py --prefix "clean_parse_10M_data_small" --model_init "../../babylm-models/baby_small_graminduct/outputs/model_best.pth.tar" --train_data "../../data/all_train_10M_data_split.txt" --val_data "../../data/all_dev_data_split.txt" --save_model_path "../../babylm-models/baby_small_graminduct52722/" --logger_name "../../babylm-models/baby_small_graminduct52722/" --batch_size 16 --val_step 5000 --seed 52722 --log_step 5000

#python ./train_lm.py --prefix "clean_parse_10M_data_small" --model_init "../../babylm-models/baby_small_graminduct/outputs/model_best.pth.tar" --train_data "../../data/all_train_10M_data_split.txt" --val_data "../../data/all_dev_data_split.txt" --save_model_path "../../babylm-models/baby_small_graminduct527/" --logger_name "../../babylm-models/baby_small_graminduct527/" --batch_size 20 --val_step 5000 --seed 527 --log_step 5000

python ./train_lm.py --prefix "clean_parse_10M_data_small" --model_init "../../babylm-models/baby_small_graminduct/outputs/model_best.pth.tar" --train_data "../../data/all_train_10M_data_split.txt" --val_data "../../data/all_dev_data_split.txt" --save_model_path "../../babylm-models/baby_small527/" --logger_name "../../babylm-models/baby_small527/" --batch_size 20 --val_step 5000 --seed 527 --log_step 5000
