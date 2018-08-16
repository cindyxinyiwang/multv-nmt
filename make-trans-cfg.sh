

TEMP_DIR=scripts/trans_template/
CFG_DIR=scripts/trans_cfg/

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

#ILS=(bel)
#RLS=(rus)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  RL=${RLS[$i]}
  echo $IL
  for f in $TEMP_DIR/*; do
    sed "s/IL/$IL/g; s/RL/$RL/g" < $f > ${f/trans_template/trans_cfg/}_$IL+$RL.sh 
    chmod u+x ${f/trans_template/trans_cfg/}_$IL+$RL.sh 
  done
done
