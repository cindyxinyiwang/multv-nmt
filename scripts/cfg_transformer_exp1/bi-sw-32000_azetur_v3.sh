#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

python src/main.py \
  --model_type='transformer' \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_transformer_exp1/bi-sw-32000_azetur_v3/" \
  --data_path data/azetur_eng/ \
  --train_src_file_list data/azetur_eng/ted-train.mtok.sepspm32000.azetur \
  --train_trg_file_list  data/azetur_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file  data/aze_eng/ted-dev.mtok.spm32000.aze \
  --dev_trg_file  data/aze_eng/ted-dev.mtok.spm8000.eng \
  --dev_trg_ref  data/aze_eng/ted-dev.mtok.eng \
  --src_vocab_list  data/azetur_eng/ted-train.mtok.sepspm32000.azetur.vocab \
  --trg_vocab_list  data/azetur_eng/ted-train.mtok.spm8000.eng.vocab \
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
  --cuda \
  --batcher='word' \
  --batch_size 4500 \
  --valid_batch_size=7 \
  --patience 10 \
  --lr_min 0.0001 \
  --lr_max 0.001 \
  --lr 0.001 \
  --lr_dec_steps 10000 \
  --n_warm_ups 2000 \
  --dropout 0.3 \
  --transformer_wdrop \
  --init_range 0.04 \
  --max_len 10000 \
  --seed 0
