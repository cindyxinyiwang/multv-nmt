#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3.6 src/main.py \
  --model_type='transformer' \
  --clean_mem_every 2 \
  --reset_output_dir \
  --char_ngram_n 5 \
  --n 5 \
  --semb='dot_prod' \
  --semb_vsize=10000 \
  --sep_char_proj \
  --trg_no_char \
  --query_base \
  --ordered_char_dict \
  --output_dir="outputs_transformer_exp1/bi-semb-o32000_azetur_v1/" \
  --data_path data/azetur_eng/ \
  --train_src_file_list data/azetur_eng/ted-train.mtok.sepspm32000.azetur \
  --train_trg_file_list  data/azetur_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file  data/aze_eng/ted-dev.mtok.spm32000.aze \
  --dev_trg_file  data/aze_eng/ted-dev.mtok.spm8000.eng \
  --dev_trg_ref  data/aze_eng/ted-dev.mtok.eng \
  --src_vocab_list  data/azetur_eng/ted-train.mtok.sepspm32000.azetur.vocab \
  --trg_vocab_list  data/azetur_eng/ted-train.mtok.spm8000.eng.vocab \
  --src_char_vocab_from  data/aze_eng/ted-train.mtok.aze.ochar5vocab,data/tur_eng/ted-train.mtok.tur.ochar5vocab \
  --trg_char_vocab_from  data/azetur_eng/ted-train.mtok.spm8000.eng.ochar5vocab \
  --src_char_vocab_size='32000,32000' \
  --trg_char_vocab_size='8000' \
  --d_word_vec=288 \
  --d_model=288 \
  --d_inner=507 \
  --n_layers=5 \
  --n_heads=3 \
  --share_emb_softmax \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=15 \
  --merge_bpe \
  --eval_bleu \
  --cuda \
  --batcher='word' \
  --batch_size 2250 \
  --update_batch 2 \
  --valid_batch_size=7 \
  --patience 10 \
  --lr_dec 0.95 \
  --lr 0.001 \
  --n_warm_ups 2000 \
  --dropout 0.3 \
  --transformer_wdrop \
  --init_range 0.04 \
  --max_len 1000 \
  --seed 0
