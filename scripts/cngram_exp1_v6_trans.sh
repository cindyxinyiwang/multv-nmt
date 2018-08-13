#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"

python3.6 src/translate.py \
  --model_dir="outputs_cngram_exp1_v6/" \
  --data_path data/aze+tur_eng/ \
  --test_src_file ted-test.orig.piece.aze \
  --test_trg_file ted-test.piece.eng \
  --src_vocab_list bi.orig.piece.vocab \
  --trg_vocab_list bi.piece.eng.vocab \
  --cuda \
  --merge_bpe \
  --beam_size=5 \
  --max_len=200 \
  --out_file="beam5" 
