from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import _pickle as pickle
import shutil
import gc
import os
import sys
import time

import numpy as np

from data_utils import DataUtil
from hparams import *
from utils import *
from model import *

import torch
import torch.nn as nn
from torch.autograd import Variable

class TranslationHparams(HParams):
  dataset = "Translate dataset"

parser = argparse.ArgumentParser(description="Neural MT translator")

parser.add_argument("--cuda", action="store_true", help="GPU or not")
parser.add_argument("--data_path", type=str, default=None, help="path to all data")
parser.add_argument("--model_dir", type=str, default="outputs", help="root directory of saved model")
parser.add_argument("--test_src_file", type=str, default=None, help="name of source test file")
parser.add_argument("--test_trg_file", type=str, default=None, help="name of target test file")
parser.add_argument("--beam_size", type=int, default=None, help="beam size")
parser.add_argument("--max_len", type=int, default=300, help="maximum len considered on the target side")
parser.add_argument("--poly_norm_m", type=float, default=0, help="m in polynormial normalization")
parser.add_argument("--non_batch_translate", action="store_true", help="use non-batched translation")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--merge_bpe", action="store_true", help="")
parser.add_argument("--src_vocab_list", type=str, default=None, help="name of source vocab file")
parser.add_argument("--trg_vocab_list", type=str, default=None, help="name of target vocab file")
parser.add_argument("--n_train_sents", type=int, default=None, help="max number of training sentences to load")
parser.add_argument("--out_file", type=str, default="trans", help="output file for hypothesis")
parser.add_argument("--debug", action="store_true", help="output file for hypothesis")

parser.add_argument("--nbest", action="store_true", help="whether to return the nbest list")
args = parser.parse_args()

model_file_name = os.path.join(args.model_dir, "model.pt")
if not args.cuda:
  model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
else:
  model = torch.load(model_file_name)
model.eval()

out_file = os.path.join(args.model_dir, args.out_file)
print("writing translation to " + out_file)
hparams = TranslationHparams(
  data_path=args.data_path,
  src_vocab_list=args.src_vocab_list,
  trg_vocab_list=args.trg_vocab_list,
  test_src_file = args.test_src_file,
  test_trg_file = args.test_trg_file,
  cuda=args.cuda,
  beam_size=args.beam_size,
  max_len=args.max_len,
  batch_size=args.batch_size,
  merge_bpe=args.merge_bpe,
  out_file=out_file,
  nbest=args.nbest,
  decode=True
)
 
#hparams.add_param("pad_id", model.hparams.pad_id)
#hparams.add_param("bos_id", model.hparams.bos_id)
#hparams.add_param("eos_id", model.hparams.eos_id)
#hparams.add_param("unk_id", model.hparams.unk_id)
model.hparams.cuda = hparams.cuda
data = DataUtil(hparams=hparams, decode=True)
filts = [model.hparams.pad_id, model.hparams.eos_id, model.hparams.bos_id]

#hparams.add_param("filtered_tokens", set(filts))
if args.debug:
  hparams.add_param("target_word_vocab_size", data.target_word_vocab_size)
  hparams.add_param("target_rule_vocab_size", data.target_rule_vocab_size)
  crit = get_criterion(hparams)

out_file = open(hparams.out_file, 'w', encoding='utf-8')

end_of_epoch = False
num_sentences = 0

x_test = data.test_x
if args.debug:
  y_test = data.test_y
else:
  y_test = None
#print(x_test)
hyps = model.translate(
      x_test, beam_size=args.beam_size, max_len=args.max_len, poly_norm_m=args.poly_norm_m)

if args.debug:
  forward_scores = []
  while not end_of_epoch:
    ((x_test, x_mask, x_len, x_count),
     (y_test, y_mask, y_len, y_count),
     batch_size, end_of_epoch) = data.next_test(test_batch_size=hparams.batch_size, sort_by_x=True)
  
    num_sentences += batch_size
    logits = model.forward(x_test, x_mask, x_len, y_test[:,:-1,:], y_mask[:,:-1], y_len, y_test[:,1:,2])
    logits = logits.view(-1, hparams.target_rule_vocab_size+hparams.target_word_vocab_size)
    labels = y_test[:,1:,0].contiguous().view(-1)
    val_loss, val_acc, rule_loss, word_loss, eos_loss, rule_count, word_count, eos_count =  \
        get_performance(crit, logits, labels, hparams, sum_loss=False)
    print("train forward:", val_loss.data)
    print("train label:", labels.data)
    logit_score = []
    for i,l in enumerate(labels): logit_score.append(logits[i][l].data[0])
    print("train_logit", logit_score)
    #print("train_label", labels)
    forward_scores.append(val_loss.sum().data[0])
    # The normal, correct way:
    #hyps = model.translate(
    #      x_test, x_len, beam_size=args.beam_size, max_len=args.max_len)
    # For debugging:
    # model.debug_translate_batch(
    #   x_test, x_mask, x_pos_emb_indices, hparams.beam_size, hparams.max_len,
    #   y_test, y_mask, y_pos_emb_indices)
    # sys.exit(0)
  print("translate_score:", sum(scores))
  print("forward_score:", sum(forward_scores))
  exit(0)

if args.nbest:
  for h_list in hyps:
    for h in h_list:
      h_best_words = map(lambda wi: data.trg_i2w_list[0][wi],
                       filter(lambda wi: wi not in filts, h))
      if hparams.merge_bpe:
        line = ''.join(h_best_words)
        line = line.replace('▁', ' ')
      else:
        line = ' '.join(h_best_words)
      line = line.strip()
      out_file.write(line + '\n')
      out_file.flush()
    out_file.write('\n')
else:
  for h in hyps:
    h_best_words = map(lambda wi: data.trg_i2w_list[0][wi],
                     filter(lambda wi: wi not in filts, h))
    if hparams.merge_bpe:
      line = ''.join(h_best_words)
      line = line.replace('▁', ' ')
    else:
      line = ' '.join(h_best_words)
    line = line.strip()
    out_file.write(line + '\n')
    out_file.flush()

print("Translated {0} sentences".format(num_sentences))
sys.stdout.flush()

out_file.close()
