#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="1"

python src/main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --char_ngram_n 4 \
  --semb="dot_prod" \
  --sep_char_proj \
  --query_base \
  --semb_vsize 10000 \
  --output_dir="outputs_v6/bi-semb-bq_glgpor/" \
  --data_path data/glgpor_eng/ \
  --train_src_file_list  data/glg_eng/ted-train.mtok.glg,data/por_eng/ted-train.mtok.por \
  --train_trg_file_list  data/glg_eng/ted-train.mtok.spm8000.eng,data/por_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file  data/glg_eng/ted-dev.mtok.glg \
  --dev_trg_file  data/glg_eng/ted-dev.mtok.spm8000.eng \
  --dev_trg_ref  data/glg_eng/ted-dev.mtok.eng \
  --src_vocab_list  data/glg_eng/ted-train.mtok.glg.vocab,data/por_eng/ted-train.mtok.por.vocab \
  --trg_vocab_list  data/glgpor_eng/ted-train.mtok.spm8000.eng.vocab,data/glgpor_eng/ted-train.mtok.spm8000.eng.vocab \
  --src_char_vocab_from  data/glgpor_eng/ted-train.mtok.glgpor \
  --trg_char_vocab_from  data/glgpor_eng/ted-train.mtok.spm8000.eng \
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
  --lr_dec 0.8 \
  --dropout 0.3 \
  --max_len 10000 \
  --seed 0
