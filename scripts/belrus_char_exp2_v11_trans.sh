#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"

python3.6 src/translate.py \
    --model_dir="belrus/belrus_char_exp2_v11/" \
    --data_path data/belrus_eng/ \
    --test_src_file data/bel_eng/ted-dev.mtok.bel \
    --test_trg_file data/bel_eng/ted-dev.mtok.spm8000.eng \
    --cuda \
    --merge_bpe \
    --beam_size=5 \
    --poly_norm_m=1 \
    --max_len=200 \
    --out_file="ted-dev-b5m1"
python3.6 src/translate.py \
    --model_dir="belrus/belrus_char_exp2_v11/" \
    --data_path data/belrus_eng/ \
    --test_src_file data/bel_eng/ted-test.mtok.bel \
    --test_trg_file data/bel_eng/ted-test.mtok.spm8000.eng \
    --cuda \
    --merge_bpe \
    --beam_size=5 \
    --poly_norm_m=1 \
    --max_len=200 \
    --out_file="ted-test-b5m1"
