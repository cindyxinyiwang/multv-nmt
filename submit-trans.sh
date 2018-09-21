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
export PYTHONPATH="$(pwd)"                                                       
export CUDA_VISIBLE_DEVICES="0" 
#source activate py36
#mkdir -p outs
version=v7_abl_s1
for f in scripts/cfg_"$version"/*_trans.sh; do
  f1=`basename $f _trans.sh`
  echo "$f1 "
  if [[ ! -e outputs_"$version"/"$f1"/ted-test-b5m1 ]]; then
    if [[ ! -e outputs_"$version"/"$f1"/model.pt ]]; then
      echo "$f1 no model file"
      continue
    fi
    echo "running $f"
    hostname
    nvidia-smi
    chmod u+x $f
    ./$f
  else
    echo "already started $f"
  fi
done
