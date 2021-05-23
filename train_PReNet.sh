#!/bin/bash

# Rain100H
python train_PReNet.py --preprocess True --save_path /gdata/xiaojie/Prenet_model --data_path /gdata/xiaojie/Prenet_Dataset/train/RainTrainH --name Rain100H --epochs 1

# Rain100L
python train_PReNet.py --preprocess True --save_path /gdata/xiaojie/Prenet_model --data_path /gdata/xiaojie/Prenet_Dataset/train/RainTrainL --name Rain100L --resume --epochs 1

