#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=mpi-job
#SBATCH --output=mpi-job.%j.out

source /curc/sw/anaconda3/latest
conda activate mycustomenv

# python main.py --train_file ./data/train_titles.csv \
# --model_name anferico/bert-for-patents \
# --run_name bert_pattent_titles \
# --save_dir ./models/bert_pattent_titles \
# --epochs 20

python main.py --train_file ./data/train_glove.csv \
--model_name anferico/bert-for-patents \
--run_name bert_pattent_glove \
--save_dir ./models/bert_pattent_glove \
--epochs 20
