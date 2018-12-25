#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=signs_kaggle
#SBATCH --mail-type=END
#SBATCH --mail-user=aw2797@nyu.edu
#SBATCH --output=results.out

module purge

python main.py --network deepbase
