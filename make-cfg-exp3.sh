
TEMP_DIR=scripts/exp6_template/
# change random seed and directory name as desired
CFG_DIR=exp6/
TEMP=exp6_template/
SEED=0

mkdir -p scripts/"$CFG_DIR"
## low-resource language codes
#ILS=(
#  aze
#  bel
#  glg
#  slk)
## paired high-resource language codes
#RLS=(
#  tur
#  rus
#  por
#  ces)

# low-resource language codes
ILS=(
  aze)
# paired high-resource language codes
RLS=(
  tur)


for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  RL=${RLS[$i]}
  echo $IL
  #for f in $TEMP_DIR/sw-8000 $TEMP_DIR/semb-8000  ; do
  #for f in $TEMP_DIR/sw-8000 ; do
  #  sed "s/VERSION/$VERSION/g; s/SEED/$SEED/g; s/IL/$IL/g; s/RL/$RL/g" < $f > ${f/"$TEMP"/"$CFG_DIR"/}_$IL$RL.sh 
  #  chmod u+x ${f/"$TEMP"/"$CFG_DIR"/}_$IL$RL.sh 
  #done
done


for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  RL=${RLS[$i]}
  echo $IL
  #for f in $TEMP_DIR/sw-8000-sam $TEMP_DIR/sw-8000-sel $TEMP_DIR/sw-8000-all $TEMP_DIR/semb-8000-sel  $TEMP_DIR/semb-8000-sam  $TEMP_DIR/semb-8000-sel ; do
  #for f in $TEMP_DIR/sw-8000-sam $TEMP_DIR/sw-8000-sel $TEMP_DIR/sw-8000-all ; do
  #for f in $TEMP_DIR/sw-8000-sam $TEMP_DIR/sw-8000-sel; do
  #for f in $TEMP_DIR/semb-8000-sam $TEMP_DIR/semb-8000-sel ; do
  for f in $TEMP_DIR/sw-8000-lm-am $TEMP_DIR/semb-8000-lm-am $TEMP_DIR/semb-8000-lm-t0.1 $TEMP_DIR/sw-8000-lm-t0.1 $TEMP_DIR/sw-8000-lm-t0.05 $TEMP_DIR/semb-8000-lm-t0.05; do
  #for f in $TEMP_DIR/semb-8000-sam-t2 $TEMP_DIR/semb-8000-sam-t10; do
    sed "s/VERSION/$VERSION/g; s/SEED/$SEED/g; s/IL/$IL/g; s/RL/$RL/g" < $f > ${f/"$TEMP"/"$CFG_DIR"/}_$IL.sh 
    chmod u+x ${f/"$TEMP"/"$CFG_DIR"/}_$IL.sh 
  done
done
