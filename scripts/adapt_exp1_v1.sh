#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3.6 src/main.py \
  --clean_mem_every 5 \
  --pretrained_model="outputs_concat_exp1_v6/model.pt" \
  --output_dir="outputs_adapt_exp1_v1/" \
  --data_path data/aze_eng/ \
  --train_src_file_list ted-train.orig.piece.aze \
  --train_trg_file_list  ted-train.piece.eng \
  --dev_src_file ted-dev.orig.piece.aze \
  --dev_trg_file ted-dev.piece.eng \
  --dev_trg_ref ted-dev.mtok.eng \
  --eval_bleu \
  --merge_bpe \
  --dev_zero \
  --ppl_thresh 100 \
  --src_vocab_list ted-train.orig.piece.aze.vocab \
  --trg_vocab_list bi.piece.eng.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=200 \
  --batcher='word' \
  --batch_size 2000 \
  --valid_batch_size=7 \
  --n_train_steps 10000 \
  --cuda \
  --dropout 0.3 \
  --max_len 60 \
  --seed 0
