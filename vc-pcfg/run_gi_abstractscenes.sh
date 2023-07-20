#!/bin/bash

#python ./train.py

#python ./train_lm.py

# stopped after 15 epochs
#python ./train.py --prefix "clean_parse_10M_data_small" --data_path "../preprocessed-data/babylm-small" --logger_name "../../babylm-models/baby_small_graminduct/outputs" 

python ./train_lm.py --prefix "clean_parse_10M_data_small" --model_init "../../babylm-models/baby_small_graminduct/outputs/model_best.pth.tar" --train_data "../../data/all_train_10M_data_split.txt" --val_data "../../data/all_dev_data_split.txt" --save_model_path "../../babylm-models/baby_small_graminduct/" --logger_name "../../babylm-models/baby_small_graminduct/" --batch_size 16 --val_step 5000
