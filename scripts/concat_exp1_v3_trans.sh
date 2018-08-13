#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3.6 src/translate.py \
  --model_dir="outputs_concat_exp1_v3/" \
  --data_path data/aze+tur+eng_eng/ \
  --test_src_file ted-test.piece.aze \
  --test_trg_file ted-test.piece.eng \
  --src_vocab_list tri.piece.vocab \
  --trg_vocab_list eng.piece.vocab \
  --cuda \
  --merge_bpe \
  --beam_size=8 \
  --max_len=200 \
  --out_file="beam8" 
