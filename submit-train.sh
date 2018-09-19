#!/bin/bash
##SBATCH --nodelist=compute-0-7
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00
#SBATCH --nodes=1
#SBATCH --mem=12g
#SBATCH --job-name="multlin"
##SBATCH --mail-user=gneubig@cs.cmu.edu
##SBATCH --mail-type=ALL
##SBATCH --requeue
#Specifies that the job will be requeued after a node failure.
#The default is that the job will not be requeued.
set -e
#export PYTHONPATH="$(pwd)"                                                       
#export CUDA_VISIBLE_DEVICES="0" 
version=7
#source activate py36
mkdir -p outputs_v7_s1
for f in scripts/cfg_v7_s1/*; do
  f1=`basename $f .sh`
  if [[ ! -e outputs_v7_s1/$f1.started ]]; then
    echo "running $f1"
    touch outputs_v7_s1/$f1.started
    hostname
    nvidia-smi
    ./$f
  else
    echo "already started $f1"
  fi
done
