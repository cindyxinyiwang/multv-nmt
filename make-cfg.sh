

TEMP_DIR=scripts/template/
CFG_DIR=scripts/cfg/

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
  echo $IL
  for f in $TEMP_DIR/*; do
    sed "s/IL/$IL/g; s/RL/$RL/g" < $f > ${f/template/cfg/}_$IL+$RL.sh 
    chmod u+x ${f/template/cfg/}_$IL+$RL.sh 
  done
done
