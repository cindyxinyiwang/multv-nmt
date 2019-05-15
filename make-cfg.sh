
TEMP_DIR=scripts/dds_template/
# change random seed and directory name as desired
CFG_DIR=dds_baseline/
SEED=0
DATA_DIR='\/home\/xinyiw\/multv-nmt\/'

mkdir -p scripts/"$CFG_DIR"
# low-resource language codes
ILS=(
  aze
  bel
  glg
  slk)
# paired high-resource language codes
RLS=(
  tur
  rus
  por
  ces)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  RL=${RLS[$i]}
  echo $IL
  for f in $TEMP_DIR/sw-8000 ; do
    sed "s/DATA_DIR/$DATA_DIR/g; s/SEED/$SEED/g; s/IL/$IL/g; s/RL/$RL/g" < $f > ${f/dds_template/"$CFG_DIR"/}_$IL$RL$SEED.sh 
    chmod u+x ${f/dds_template/"$CFG_DIR"/}_$IL$RL$SEED.sh 
  done
done
