#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0

python src/main.py \
  --output_dir="outputs_concat_exp1_v1/" \
  --data_path data/aze+tur_eng/ \
  --train_src_file_list bi.piece.bi \
  --train_trg_file_list  bi.piece.eng \
  --dev_src_file ted-dev.piece.aze \
  --dev_trg_file ted-dev.piece.eng \
  --src_vocab_list bi.piece.bi.vocab \
  --trg_vocab_list bi.piece.eng.vocab \
  --d_word_vec=512 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=1500 \
  --batcher='word' \
  --batch_size 2000 \
  --valid_batch_size=7 \
  --n_train_steps 50000 \
  --cuda \
  --seed 0
