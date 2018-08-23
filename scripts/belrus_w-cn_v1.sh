#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

python src/main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --char_ngram_n 4 \
  --src_vocab_size 40000 \
  --output_dir="belrus/belrus_w-cn_v1/" \
  --data_path data/belrus_eng/ \
  --train_src_file_list data/belrus_eng/ted-train.mtok.belrus \
  --train_trg_file_list  data/belrus_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file  data/bel_eng/ted-dev.mtok.bel \
  --dev_trg_file  data/bel_eng/ted-dev.mtok.spm8000.eng \
  --dev_trg_ref  data/bel_eng/ted-dev.mtok.eng \
  --src_vocab_list  data/belrus_eng/ted-train.mtok.belrus.vocab \
  --trg_vocab_list  data/belrus_eng/ted-train.mtok.spm8000.eng.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=13 \
  --merge_bpe \
  --eval_bleu \
  --batcher='word' \
  --batch_size 1500 \
  --valid_batch_size=7 \
  --patience 5 \
  --lr_dec 0.95 \
  --dropout 0.3 \
  --max_len 10000 \
  --cuda \
  --seed 0
