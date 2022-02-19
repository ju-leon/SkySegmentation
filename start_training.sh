#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128gb
#SBATCH --time=18:00:00


python train.py --data_dir='/pfs/work7/workspace/scratch/utpqw-skyseg' \
		--save_dir='log' \
		--num_classes=5 \
		--epochs=60 \
		--eval_dir='eval'
