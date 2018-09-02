#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="0"

python src/main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --char_ngram_n 4 \
  --semb="dot_prod" \
  --sep_char_proj \
  --query_base \
  --char_gate \
  --semb_vsize 10000 \
  --output_dir="outputs_v6/bi-semb-bq-all-v1_belrus/" \
  --data_path data/belrus_eng/ \
  --train_src_file_list  data/bel_eng/ted-train.mtok.bel,data/rus_eng/ted-train.mtok.rus,data/tur_eng/ted-train.mtok.tur,data/por_eng/ted-train.mtok.por,data/ces_eng/ted-train.mtok.ces \
  --train_trg_file_list  data/bel_eng/ted-train.mtok.spm8000.eng,data/rus_eng/ted-train.mtok.spm8000.eng,data/tur_eng/ted-train.mtok.spm8000.eng,data/por_eng/ted-train.mtok.spm8000.eng,data/ces_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file  data/bel_eng/ted-dev.mtok.bel \
  --dev_trg_file  data/bel_eng/ted-dev.mtok.spm8000.eng \
  --dev_trg_ref  data/bel_eng/ted-dev.mtok.eng \
  --src_vocab_list  data/bel_eng/ted-train.mtok.bel.vocab,data/rus_eng/ted-train.mtok.rus.vocab,data/tur_eng/ted-train.mtok.tur.vocab,data/por_eng/ted-train.mtok.por.vocab,data/ces_eng/ted-train.mtok.ces.vocab \
  --trg_vocab_list  data/belrus_eng/ted-train.mtok.spm8000.eng.vocab,data/belrus_eng/ted-train.mtok.spm8000.eng.vocab,data/belrus_eng/ted-train.mtok.spm8000.eng.vocab,data/belrus_eng/ted-train.mtok.spm8000.eng.vocab,data/belrus_eng/ted-train.mtok.spm8000.eng.vocab \
  --src_char_vocab_from  data/bel_eng/ted-train.mtok.bel.char4vocab,data/rus_eng/ted-train.mtok.rus.char4vocab,data/tur_eng/ted-train.mtok.tur.char4vocab,data/por_eng/ted-train.mtok.por.char4vocab,data/ces_eng/ted-train.mtok.ces.char4vocab \
  --src_char_vocab_size='20000,50000,8000,8000,8000' \
  --trg_char_vocab_from  data/belrus_eng/ted-train.mtok.spm8000.eng.char4vocab \
  --trg_char_vocab_size='-1' \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=15 \
  --merge_bpe \
  --eval_bleu \
  --cuda \
  --batcher='word' \
  --batch_size 1500 \
  --valid_batch_size=7 \
  --patience 5 \
  --lr_dec 1 \
  --dropout 0.3 \
  --max_len 10000 \
  --seed 0
