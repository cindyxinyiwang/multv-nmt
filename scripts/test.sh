

python3.6 src/main.py \
	--data_path data/test/ \
	--train_src_file_list head.ja,head.ja \
	--train_trg_file_list head.en,head.en \
	--dev_src_file head.ja \
	--dev_trg_file head.en \
	--src_vocab_list vocab.ja,vocab.ja \
	--trg_vocab_list vocab.en,vocab.en \
	--batch_size 2 \
	--valid_batch_size 2 \
	--log_every 1 \
	--eval_every 2 \
  --max_char_vocab_size 60 \
  --cuda \
  --char_ngram_n 4 \
	--n_train_steps 200
