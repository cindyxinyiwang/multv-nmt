#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3.6 src/main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --char_ngram_n 4 \
  --semb='zero' \
  --output_dir="belrus/belrus_char_exp2_v6/" \
  --data_path data/belrus_eng/ \
  --train_src_file_list data/bel_eng/ted-train.mtok.bel,data/rus_eng/ted-train.mtok.rus \
  --train_trg_file_list  data/bel_eng/ted-train.mtok.spm8000.eng,data/rus_eng/ted-train.mtok.spm8000.eng  \
  --dev_src_file  data/bel_eng/ted-dev.mtok.bel \
  --dev_trg_file  data/bel_eng/ted-dev.mtok.spm8000.eng \
  --dev_trg_ref  data/bel_eng/ted-dev.mtok.eng \
  --src_vocab_list  data/bel_eng/ted-train.mtok.bel.vocab,data/rus_eng/ted-train.mtok.rus.vocab \
  --trg_vocab_list  data/belrus_eng/ted-train.mtok.spm8000.eng.vocab,data/belrus_eng/ted-train.mtok.spm8000.eng.vocab \
  --src_char_vocab_from data/belrus_eng/ted-train.mtok.belrus \
  --trg_char_vocab_from data/belrus_eng/ted-train.mtok.spm8000.eng \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=13 \
  --merge_bpe \
  --eval_bleu \
  --cuda \
  --batcher='word' \
  --batch_size 1500 \
  --valid_batch_size=7 \
  --patience 5 \
  --lr_dec 0.8 \
  --dropout 0.3 \
  --max_len 10000 \
  --seed 0
