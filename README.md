# multv-nmt

## data
put data under a folder named data/

## make config files for training
Change the directory name, random seed, language code as desired in make-cfg.sh, then run
`./make-cfg.sh`
The config files are written under the folder scripts/

## Decode
Make appropriate changes to make-trans.sh, then run
`python make-trans.py`
The translation config files will be written to the corresponding directory under scripts/
This will only create translation config for the output folder that does not contain a translation already.
