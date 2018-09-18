#!/bin/bash

BASE=/projects/tir1/users/xinyiw1/
CDIR=$BASE/usr/local/cdec
MDIR=$BASE/usr/local/mosesdecoder
XDIR=$BASE/multv-nmt
UDIR=$BASE/utils
DDIR=/projects/tir1/corpora/multiling-text
IL=bel
RL=rus
DATA_PRE=ted
vocab_size=500

mkdir -p  data/"$IL"_eng
mkdir -p  data/"$RL"_eng
mkdir -p  data/"$IL$RL"_eng

#ln -s $DDIR/ted/"$IL"_eng/ted-{train,dev,test}.orig."$IL"-eng data/"$IL"_eng/
#ln -s $DDIR/ted/"$RL"_eng/ted-{train,dev,test}.orig."$RL"-eng data/"$RL"_eng/
#
#for f in data/"$IL"_eng/*.orig.*-eng  data/"$RL"_eng/*.orig.*-eng; do
#  src=`echo $f | sed 's/-eng$//g'`
#  trg=`echo $f | sed 's/\.[^\.]*$/.eng/g'`
#  echo "src=$src, trg=$trg"
#  python cut-corpus.py 0 < $f > $src
#  python cut-corpus.py 1 < $f > $trg
#done
#
#for f in data/"$IL"_eng/*.orig.{eng,$IL}  data/"$RL"_eng/*.orig.{eng,$RL}; do
#  f1=${f/orig/mtok}
#  #cat $f | perl $MDIR/scripts/tokenizer/tokenizer.perl > $f1
#  cat $f | python src/reversible_tokenize.py > $f1
#done
#
#for split in train; do
#  cat data/"$IL"_eng/$DATA_PRE-$split.orig."$IL" data/"$RL"_eng/$DATA_PRE-$split.orig."$RL"  > data/"$IL$RL"_eng/$DATA_PRE-$split.orig.$IL$RL
#  cat data/"$IL"_eng/$DATA_PRE-$split.mtok."$IL" data/"$RL"_eng/$DATA_PRE-$split.mtok."$RL"  > data/"$IL$RL"_eng/$DATA_PRE-$split.mtok.$IL$RL
#  cat data/"$IL"_eng/$DATA_PRE-$split.orig.eng data/"$RL"_eng/$DATA_PRE-$split.orig.eng  > data/"$IL$RL"_eng/$DATA_PRE-$split.orig.eng
#  cat data/"$IL"_eng/$DATA_PRE-$split.mtok.eng data/"$RL"_eng/$DATA_PRE-$split.mtok.eng  > data/"$IL$RL"_eng/$DATA_PRE-$split.mtok.eng
#done
#
#for split in dev test; do
#  cp data/"$IL"_eng/$DATA_PRE-$split.orig.$IL data/"$IL$RL"_eng/
#  cp data/"$IL"_eng/$DATA_PRE-$split.mtok.$IL data/"$IL$RL"_eng/
#  cp data/"$IL"_eng/$DATA_PRE-$split.orig.eng data/"$IL$RL"_eng/
#  cp data/"$IL"_eng/$DATA_PRE-$split.mtok.eng data/"$IL$RL"_eng/
#  cp data/"$RL"_eng/$DATA_PRE-$split.orig.$RL data/"$IL$RL"_eng/
#  cp data/"$RL"_eng/$DATA_PRE-$split.mtok.$RL data/"$IL$RL"_eng/
#  cp data/"$RL"_eng/$DATA_PRE-$split.orig.eng data/"$IL$RL"_eng/
#  cp data/"$RL"_eng/$DATA_PRE-$split.mtok.eng data/"$IL$RL"_eng/
#done

echo "train spm from data/'$IL$RL'_eng/$DATA_PRE-train.mtok.'$IL$RL'"
python $UDIR/train-spm.py \
  --input=data/"$IL$RL"_eng/$DATA_PRE-train.mtok."$IL$RL" \
  --model_prefix=data/"$IL$RL"_eng/spm"$vocab_size.mtok.$IL$RL" \
  --vocab_size="$vocab_size" 

echo "train spm from data/'$IL'_eng/'$DATA_PRE'-train.mtok.'$IL'"
python $UDIR/train-spm.py \
  --input=data/"$IL"_eng/"$DATA_PRE"-train.mtok."$IL" \
  --model_prefix=data/"$IL"_eng/spm"$vocab_size.mtok.$IL" \
  --vocab_size="$vocab_size"

echo "train spm from data/'$RL'_eng/'$DATA_PRE'-train.mtok.'$RL'"
python $UDIR/train-spm.py \
  --input=data/"$RL"_eng/"$DATA_PRE"-train.mtok."$RL" \
  --model_prefix=data/"$RL"_eng/spm"$vocab_size.mtok.$RL" \
  --vocab_size="$vocab_size"

for f in data/"$IL"_eng/*.mtok.eng data/"$RL"_eng/*.mtok.eng data/"$IL$RL"_eng/*.mtok.eng; 
do
  python $UDIR/run-spm.py \
    --model=data/eng/spm"$vocab_size".mtok.eng.model \
    < $f \
    > ${f/mtok/mtok.spm$vocab_size} 
done

for f in data/"$IL"_eng/*.mtok."$IL"; 
do
  python $UDIR/run-spm.py \
    --model=data/"$IL"_eng/spm"$vocab_size.mtok.$IL".model \
    < $f \
    > ${f/mtok/mtok.spm$vocab_size} 
done

for f in data/"$RL"_eng/*.mtok."$RL"; 
do
  python $UDIR/run-spm.py \
    --model=data/"$RL"_eng/spm"$vocab_size.mtok.$RL".model \
    < $f \
    > ${f/mtok/mtok.spm$vocab_size} 
done

for f in data/"$IL$RL"_eng/*.mtok.{"$IL$RL","$IL"}; 
do
  python $UDIR/run-spm.py \
    --model=data/"$IL$RL"_eng/spm"$vocab_size.mtok.$IL$RL".model \
    < $f \
    > ${f/mtok/mtok.spm$vocab_size} 
done

cat data/"$IL"_eng/$DATA_PRE-train.mtok.spm$vocab_size."$IL" data/"$RL"_eng/$DATA_PRE-train.mtok.spm$vocab_size."$RL"  > data/"$IL$RL"_eng/$DATA_PRE-train.mtok.sepspm$vocab_size.$IL$RL

#for f in data/"$IL"_eng/*train.mtok.* data/"$RL"_eng/*train*.mtok.* data/"$IL$RL"_eng/$DATA_PRE-train.mtok.{eng,"$IL$RL"} data/"$IL"_eng/*train.mtok.spm8000.* data/"$RL"_eng/*train*.mtok.spm8000.* data/"$IL$RL"_eng/$DATA_PRE-train.mtok.spm8000.{eng,$IL$RL} data/"$IL$RL"_eng/$DATA_PRE-train.mtok.sepspm8000.$IL$RL; do
for f in data/"$IL"_eng/*train.mtok.spm$vocab_size.* data/"$RL"_eng/*train*.mtok.spm$vocab_size.* data/"$IL$RL"_eng/$DATA_PRE-train.mtok.spm$vocab_size.{eng,$IL$RL} data/"$IL$RL"_eng/$DATA_PRE-train.mtok.sepspm$vocab_size.$IL$RL; do
  echo "python $XDIR/src/get_vocab.py < $f > $f.vocab &"
  python $XDIR/src/get_vocab.py < $f > $f.vocab &
done

#for f in data/"$IL"_eng/*train.orig.* data/"$RL"_eng/*train*.orig.* data/"$IL"+"$RL"_eng/bi.orig.bi data/"$IL"+"$RL"_eng/bi.orig.piece.bi; do
#  echo "python $XDIR/src/get_vocab.py < $f > $f.vocab &"
#  python $XDIR/src/get_vocab.py < $f > $f.vocab &
#done
#
wait
