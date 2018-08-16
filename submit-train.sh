#!/bin/bash
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

source activate py36
mkdir -p outputs_v3
for f in scripts/cfg/*; do
  f1=`basename $f .sh`
  if [[ ! -e outputs_v3/$f1.started ]]; then
    echo "running $f1"
    touch outputs_v3/$f1.started
    hostname
    nvidia-smi
    ./$f
  else
    echo "already started $f1"
  fi
done
