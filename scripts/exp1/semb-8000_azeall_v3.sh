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
  --semb='dot_prod' \
  --semb_vsize=10000 \
  --trg_no_char \
  --ordered_char_dict \
  --sep_char_proj \
  --output_dir="outputs_exp1/semb-8000_azeall_v3/" \
  --train_src_file_list data/LAN_eng/ted-train.mtok.LAN \
  --train_trg_file_list  data/LAN_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file_list  data/aze_eng/ted-dev.mtok.aze \
  --dev_trg_file_list  data/aze_eng/ted-dev.mtok.spm8000.eng \
  --dev_ref_file_list  data/aze_eng/ted-dev.mtok.eng \
  --dev_file_idx_list  "0" \
  --src_char_vocab_from  data/LAN_eng/ted-train.mtok.LAN.char4vocab \
  --src_char_vocab_size='8000' \
  --trg_vocab  data/eng/ted-train.mtok.spm8000.eng.vocab \
  --lang_file lang_azeall.txt \
  --d_word_vec=288 \
  --d_model=288 \
  --d_inner=512 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=15 \
  --merge_bpe \
  --eval_bleu \
  --batcher='word' \
  --batch_size 2500 \
  --valid_batch_size=7 \
  --lr_dec 0.8 \
  --lr 0.001 \
  --patience 5 \
  --dropout 0.3 \
  --select_data \
  --max_len 10000 \
  --cuda \
  --seed 0
