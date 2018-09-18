#!/bin/bash

BASE=/projects/tir1/users/xinyiw1/
CDIR=$BASE/usr/local/cdec
MDIR=$BASE/usr/local/mosesdecoder
XDIR=$BASE/multv-nmt
UDIR=$BASE/utils
DDIR=/projects/tir1/corpora/multiling-text
DATA_PRE=ted
vocab_size=32000


ILS=(
  aze
  bel
  glg
  slk)
RLS=(
  tur
  rus
  por
  ces)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  RL=${RLS[$i]}
  #for f in data/"$IL"_eng/ted-train.mtok."$IL" data/"$RL"_eng/ted-train.mtok."$RL" data/"$IL$RL"_eng/ted-train.mtok."$IL$RL" data/"$IL$RL"_eng/ted-train.mtok.spm8000.eng; do
  for f in data/"$IL"_eng/ted-train.mtok.spm$vocab_size."$IL" data/"$RL"_eng/ted-train.mtok.spm$vocab_size."$RL" data/"$IL$RL"_eng/ted-train.mtok.spm$vocab_size."$IL$RL"; do
    echo "python $XDIR/src/get_char_vocab.py < $f > $f.char5vocab &"
    python $XDIR/src/get_char_vocab.py --n 5 < $f > $f.char5vocab &
    echo "python $XDIR/src/get_char_vocab.py --orderd --n 5 < $f > $f.ochar5vocab &"
    python $XDIR/src/get_char_vocab.py --ordered --n 5 < $f > $f.ochar5vocab &
  done
done
wait
