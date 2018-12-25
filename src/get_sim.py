import argparse
import pickle as pkl

from model import *
from transformer import *
from utils import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from mult_data_utils import MultDataUtil


vocab_size = 8000
base_lan = "aze"
#lan_lists = ["rus", "por", "ces"]
lan_lists = ["aze", "tur", "rus", "por", "ces"]
cuda = True

def sim_by_model(model_dir):
  model_file_name = os.path.join(model_dir, "model.pt")
  if not cuda:
    model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
  else:
    model = torch.load(model_file_name)
  model.eval()
  
  model.hparams.shuffle_train = False
  model.hparams.batcher = "sent"
  model.hparams.batch_size =  1
  model.hparams.cuda = cuda
  model.hparams.sample_select = False 
  model.hparams.sep_char_proj = False 
  crit = get_criterion(model.hparams)
  for lan in lan_lists:
    model.hparams.lang_file = "lang_{}.txt".format(lan)
    data = MultDataUtil(model.hparams, shuffle=False)
    out = open("data/{}_eng/ted-train.mtok.{}.sim-nmt".format(lan, lan), "w")
 
    step = 0
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, dev_file_index in data.next_train():
      gc.collect()
      logits = model.forward(
        x, x_mask, x_len, x_pos_emb_idxs,
        y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=dev_file_index, step=step)
      logits = logits.view(-1, model.hparams.trg_vocab_size)
      labels = y[:,1:].contiguous().view(-1)
      val_loss, val_acc = get_performance(crit, logits, labels, model.hparams, sum_loss=False)
      y_len = torch.FloatTensor(y_len)
      if cuda: y_len = y_len.cuda()
      ave_loss = val_loss.sum(-1) / y_len
      for i in range(model.hparams.batch_size):
        out.write("{}\n".format(ave_loss[i].item()))
      if eop: break
      if step == 10: break
      if step % 1000 == 0: print("{} {}".format(lan, step))
      step += 1

def sim_by_ngram(base_lan, lan_lists):
  base_vocab = "data/{}_eng/ted-train.mtok.{}.ochar4vocab".format(base_lan, base_lan)
  base_vocab_set = set([])
  with open(base_vocab, "r") as myfile:
    for line in myfile:
      base_vocab_set.add(line.strip())

  for lan in lan_lists:
    train = open("data/{}_eng/ted-train.mtok.{}".format(lan, lan), "r")
    out = open("data/{}_eng/ted-train.mtok.{}.sim-ngram".format(lan, lan), "w")
    for line in train:
      words = line.split()
      sim = 0
      for w in words:
       s = 0
       for l in range(1, len(w)):
         for i in range(len(w)-l+1):
           if w[i:i+l] in base_vocab_set: s += 1
       sim += (s / len(w))
      out.write("{}\n".format(sim / len(words)))

def sim_by_ngram_v1(base_lan, lan_lists):
  base_vocab = "data/{}_eng/ted-train.mtok.{}.ochar4vocab".format(base_lan, base_lan)
  base_vocab_set = set([])
  with open(base_vocab, "r") as myfile:
    for line in myfile:
      base_vocab_set.add(line.strip())

  for lan in lan_lists:
    train = open("data/{}_eng/ted-train.mtok.{}".format(lan, lan), "r")
    out = open("data/{}_eng/ted-train.mtok.{}.sim-ngram_v1".format(lan, lan), "w")
    for line in train:
      words = line.split()
      sim = 0
      for w in words:
       s = 0
       for l in range(1, len(w)):
         for i in range(len(w)-l+1):
           if w[i:i+l] in base_vocab_set: s += 1
       sim += s
      out.write("{}\n".format(sim / len(words)))

if __name__ == "__main__":
  sim_by_ngram_v1(base_lan, lan_lists)
  #sim_by_ngram(base_lan, lan_lists)
  #sim_by_model("outputs_exp1/semb-8000_azetur_v2/")
