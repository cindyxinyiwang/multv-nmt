#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"

python3.6 src/translate.py \
  --model_dir="outputs_transformer_exp2/bi-sw-32000_azetur_v5/" \
  --data_path data/azetur_eng/ \
  --test_src_file_list data/aze_eng/ted-test.mtok.spm32000.aze \
  --test_trg_file_list data/aze_eng/ted-test.mtok.spm8000.eng \
  --test_file_idx_list "0" \
  --cuda \
  --merge_bpe \
  --beam_size=5 \
  --poly_norm_m=0 \
  --max_len=200 \
  --out_file_list "ted-test-b5m0" 
