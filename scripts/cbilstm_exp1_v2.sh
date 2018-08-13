#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3.6 src/main.py \
  --clean_mem_every 3 \
  --char_input \
  --output_dir="outputs_cbilstm_exp1_v2/" \
  --data_path data/aze+tur_eng/ \
  --train_src_file_list bi.orig.piece.bi \
  --train_trg_file_list  bi.piece.eng \
  --dev_src_file ted-dev.orig.piece.aze \
  --dev_trg_file ted-dev.piece.eng \
  --src_vocab_list bi.orig.piece.vocab \
  --trg_vocab_list bi.piece.eng.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=1500 \
  --batcher='word' \
  --batch_size 2000 \
  --valid_batch_size=7 \
  --n_train_steps 100000 \
  --cuda \
  --dropout 0.3 \
  --max_len 60 \
  --seed 0
