#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3.6 src/main_v2.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --char_ngram_n 4 \
  --semb='dot_prod' \
  --semb_vsize=10000 \
  --trg_no_char \
  --ordered_char_dict \
  --sep_char_proj \
  --mask_weight 0.5 \
  --output_dir="outputs_exp1/semb-8000_azeall_w_v1/" \
  --train_src_file_list data/LAN_eng/ted-train.mtok.LAN \
  --train_trg_file_list  data/LAN_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file_list  data/aze_eng/ted-dev.mtok.aze,data/tur_eng/ted-dev.mtok.tur,data/rus_eng/ted-dev.mtok.rus,data/por_eng/ted-dev.mtok.por,data/ces_eng/ted-dev.mtok.ces   \
  --dev_trg_file_list  data/aze_eng/ted-dev.mtok.spm8000.eng,data/tur_eng/ted-dev.mtok.spm8000.eng,data/rus_eng/ted-dev.mtok.spm8000.eng,data/por_eng/ted-dev.mtok.spm8000.eng,data/ces_eng/ted-dev.mtok.spm8000.eng   \
  --dev_ref_file_list  data/aze_eng/ted-dev.mtok.eng,data/tur_eng/ted-dev.mtok.eng,data/rus_eng/ted-dev.mtok.eng,data/por_eng/ted-dev.mtok.eng,data/ces_eng/ted-dev.mtok.eng   \
  --dev_file_idx_list  "0,1,2,3,4" \
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
  --max_len 10000 \
  --cuda \
  --seed 0
