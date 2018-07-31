#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0

python src/main.py \
  --output_dir="outputs_share_exp1_v1/" \
  --data_path data/ \
  --train_src_file_list aze_eng/ted-train.piece.aze,tur_eng/ted-train.piece.tur \
  --train_trg_file_list aze_eng/ted-train.piece.eng,tur_eng/ted-train.piece.eng \
  --dev_src_file aze_eng/ted-dev.piece.aze \
  --dev_trg_file aze_eng/ted-dev.piece.eng \
  --src_vocab_list aze_eng/ted-train.piece.aze.vocab,tur_eng/ted-train.piece.tur.vocab \
  --trg_vocab_list aze_eng/ted-train.piece.eng.vocab,tur_eng/ted-train.piece.eng.vocab \
  --d_word_vec=512 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=1500 \
  --batcher='word' \
  --batch_size 2000 \
  --valid_batch_size=7 \
  --n_train_steps 50000 \
  --cuda \
  --seed 0
