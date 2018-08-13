#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3.6 src/main.py \
  --output_dir="outputs_concat_exp1_v3/" \
  --data_path data/aze+tur+eng_eng/ \
  --train_src_file_list tri.piece.tri \
  --train_trg_file_list  tri.piece.eng \
  --dev_src_file ted-dev.piece.aze \
  --dev_trg_file ted-dev.piece.eng \
  --src_vocab_list tri.piece.vocab \
  --trg_vocab_list eng.piece.vocab \
  --d_word_vec=512 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=1500 \
  --batcher='word' \
  --batch_size 2000 \
  --valid_batch_size=7 \
  --n_train_steps 100000 \
  --cuda \
  --seed 0
