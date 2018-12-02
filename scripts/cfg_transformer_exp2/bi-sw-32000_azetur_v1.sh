#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3.6 src/main.py \
  --model_type='transformer' \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_transformer_exp2/bi-sw-32000_azetur_v1/" \
  --train_src_file_list data/LAN_eng/ted-train.mtok.spm32000.LAN \
  --train_trg_file_list  data/LAN_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file_list  data/aze_eng/ted-dev.mtok.spm32000.aze \
  --dev_trg_file_list  data/aze_eng/ted-dev.mtok.spm8000.eng \
  --dev_ref_file_list  data/aze_eng/ted-dev.mtok.eng \
  --dev_file_idx_list  "0" \
  --src_vocab_list  data/LAN_eng/ted-train.mtok.spm32000.LAN.vocab \
  --trg_vocab  data/eng/ted-train.mtok.spm8000.eng.vocab \
  --lang_file lang_azetur.txt \
  --d_word_vec=288 \
  --d_model=288 \
  --d_inner=512 \
  --n_layers=5 \
  --n_heads=3 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=15 \
  --merge_bpe \
  --eval_bleu \
  --batcher='word' \
  --batch_size 4500 \
  --valid_batch_size=7 \
  --patience 10 \
  --lr_dec 0.95 \
  --lr 0.001 \
  --n_warm_ups 2000 \
  --dropout 0.3 \
  --sep_step 30000 \
  --sep_relative_loc \
  --transformer_relative_pos \
  --relative_pos_c \
  --relative_pos_d \
  --init_range 0.04 \
  --transformer_wdrop \
  --max_len 10000 \
  --cuda \
  --seed 0
