import random
import numpy as np
import os

import torch
from torch.autograd import Variable

class DataUtil(object):

  def __init__(self, hparams, decode=True):
    self.hparams = hparams
    self.src_i2w_list = []
    self.src_w2i_list = []
    for v_file in hparams.src_vocab_list:
      v_file = os.path.join(self.hparams.data_path, v_file)
      i2w, w2i = self._build_vocab(v_file, max_vocab_size=self.hparams.src_vocab_size)   
      self.src_i2w_list.append(i2w)
      self.src_w2i_list.append(w2i)
      if self.hparams.src_vocab_size is None:
        self.hparams.src_vocab_size = len(i2w)
        print("setting src_vocab_size to {}...".format(self.hparams.src_vocab_size))

    self.trg_i2w_list = []
    self.trg_w2i_list = []
    for v_file in hparams.trg_vocab_list:
      v_file = os.path.join(self.hparams.data_path, v_file)
      i2w, w2i = self._build_vocab(v_file, max_vocab_size=self.hparams.trg_vocab_size)   
      self.trg_i2w_list.append(i2w)
      self.trg_w2i_list.append(w2i)
      if self.hparams.trg_vocab_size is None:
        self.hparams.trg_vocab_size = len(i2w)
        print("setting trg_vocab_size to {}...".format(self.hparams.trg_vocab_size))

    if not self.hparams.decode:
      assert len(self.src_i2w_list) == len(self.hparams.train_src_file_list)
      assert len(self.trg_i2w_list) == len(self.hparams.train_trg_file_list)
      self.train_x = []
      self.train_y = []
      i, self.train_size = 0, 0
      self.n_train_batches = None
      for s_file,t_file in zip(self.hparams.train_src_file_list, self.hparams.train_trg_file_list):
        s_file = os.path.join(self.hparams.data_path, s_file)
        t_file = os.path.join(self.hparams.data_path, t_file)
        train_x, train_y = self._build_parallel(s_file, t_file, i)
        self.train_x.extend(train_x)
        self.train_y.extend(train_y)
        i += 1
        self.train_size += len(train_x)
      if not self.hparams.load_model:
        dev_src_file = os.path.join(self.hparams.data_path, self.hparams.dev_src_file)
        dev_trg_file = os.path.join(self.hparams.data_path, self.hparams.dev_trg_file)
      else:
        dev_src_file = self.hparams.dev_src_file
        dev_trg_file = self.hparams.dev_trg_file
      self.dev_x, self.dev_y = self._build_parallel(dev_src_file, dev_trg_file, 0)
      self.dev_size = len(self.dev_x)
      self.reset_train()
      self.dev_index = 0
    else:
      test_src_file = os.path.join(self.hparams.data_path, self.hparams.test_src_file)
      test_trg_file = os.path.join(self.hparams.data_path, self.hparams.test_trg_file)
      self.test_x, self.test_y = self._build_parallel(test_src_file, test_trg_file, 0)
      self.test_size = len(self.test_x)
      self.test_index = 0

  def reset_train(self):
    if self.hparams.batcher == "word":
      if self.n_train_batches is None:
        start_indices, end_indices = [], []
        start_index = 0
        while start_index < self.train_size:
          end_index = start_index
          word_count = 0
          while (end_index + 1 < self.train_size and word_count + len(self.train_x[end_index]) + len(self.train_y[end_index]) <= self.hparams.batch_size):
            end_index += 1
            word_count += (len(self.train_x[end_index]) + len(self.train_y[end_index]))
            start_indices.append(start_index)
            end_indices.append(end_index+1)
            start_index = end_index + 1 
          assert len(start_indices) == len(end_indices)
          self.n_train_batches = len(start_indices)
          self.start_indices, self.end_indices = start_indices, end_indices
    elif self.hparams.batcher == "sent":
      if self.n_train_batches is None:
        self.n_train_batches = ((self.train_size + self.hparams.batch_size - 1) // self.hparams.batch_size)
    else:
      print("unknown batcher")
      exit(1)
    self.train_queue = np.random.permutation(self.n_train_batches)
    self.train_index = 0

  def next_train(self):
    if self.hparams.batcher == "word":
      start_index = self.start_indices[self.train_queue[self.train_index]]
      end_index = self.end_indices[self.train_queue[self.train_index]]
    elif self.hparams.batcher == "sent":
      start_index = (self.train_queue[self.train_index] * self.hparams.batch_size)
      end_index = min(start_index + self.hparams.batch_size, self.train_size)
    else:
      print("unknown batcher")
      exit(1)

    x_train = self.train_x[start_index:end_index]
    y_train = self.train_y[start_index:end_index]
    x_train, y_train = self.sort_by_xlen(x_train, y_train)

    self.train_index += 1
    batch_size = len(x_train)
    y_count = sum([len(y) for y in y_train])
    # pad 
    x_train, x_mask, x_count, x_len = self._pad(x_train, self.hparams.pad_id)
    y_train, y_mask, y_count, y_len = self._pad(y_train, self.hparams.pad_id)

    if self.train_index >= self.n_train_batches:
      self.reset_train()
    return x_train, x_mask, x_count, x_len, y_train, y_mask, y_count, y_len, batch_size

  def next_dev(self, dev_batch_size=10):
    start_index = self.dev_index
    end_index = min(start_index + dev_batch_size, self.dev_size)
    batch_size = end_index - start_index

    x_dev = self.dev_x[start_index:end_index]
    y_dev = self.dev_y[start_index:end_index]
    x_dev, y_dev = self.sort_by_xlen(x_dev, y_dev)

    x_dev, x_mask, x_count, x_len = self._pad(x_dev, self.hparams.pad_id)
    y_dev, y_mask, y_count, y_len = self._pad(y_dev, self.hparams.pad_id)

    if end_index >= self.dev_size:
      eop = True
      self.dev_index = 0
    else:
      eop = False
      self.dev_index += batch_size

    return x_dev, x_mask, x_count, x_len, y_dev, y_mask, y_count, y_len, batch_size, eop

  def next_test(self, test_batch_size=10):
    start_index = self.test_index
    end_index = min(start_index + test_batch_size, self.test_size)
    batch_size = end_index - start_index

    x_test = self.test_x[start_index:end_index]
    y_test = self.test_y[start_index:end_index]
    x_test, x_mask, x_count, x_len = self._pad(x_test, self.pad_id)
    y_test, y_mask, y_count, y_len = self._pad(y_test, self.pad_id)

    if end_index >= self.test_size:
      eop = True
      self.test_index = 0
    else:
      eop = False
      self.test_index += batch_size

    return x_test, x_mask, x_count, x_len, y_test, y_mask, y_count, y_len, batch_size, eop

  def sort_by_xlen(self, x, y):
    x = np.array(x)
    y = np.array(y)
    x_len = [len(i) for i in x]
    index = np.argsort(x_len)[::-1]
    #print(x)
    #print(y)
    #print(index)
    #print(x_len)
    x, y = x[index].tolist(), y[index].tolist()
    return x, y

  def _pad(self, sentences, pad_id):
    batch_size = len(sentences)
    lengths = [len(s) for s in sentences]
    count = sum(lengths)
    max_len = max(lengths)
    padded_sentences = [s + ([pad_id]*(max_len - len(s))) for s in sentences]
    mask = [[0]*len(s) + [1]*(max_len - len(s)) for s in sentences]
    padded_sentences = Variable(torch.LongTensor(padded_sentences))
    mask = torch.ByteTensor(mask)
    if self.hparams.cuda:
      padded_sentences = padded_sentences.cuda()
      mask = mask.cuda()
    return padded_sentences, mask, count, lengths

  def _build_parallel(self, src_file_name, trg_file_name, i):
    print("loading parallel sentences from {} {} with vocab {}".format(src_file_name, trg_file_name, i))
    with open(src_file_name, 'r', encoding='utf-8') as f:
      src_lines = f.read().split('\n')
    with open(trg_file_name, 'r', encoding='utf-8') as f:
      trg_lines = f.read().split('\n')
    src_data = []
    trg_data = []
    line_count = 0
    for src_line, trg_line in zip(src_lines, trg_lines):
      src_tokens = src_line.split()
      trg_tokens = trg_line.split()
      if not src_tokens or not trg_tokens: continue
      if not self.hparams.decode and self.hparams.max_len and len(src_tokens) > self.hparams.max_len and len(trg_tokens) > self.hparams.max_len:
        continue

      src_indices, trg_indices = [self.hparams.bos_id], [self.hparams.bos_id] 
      src_unk_count = 0
      trg_unk_count = 0
      src_w2i = self.src_w2i_list[i]
      for src_tok in src_tokens:
        if src_tok not in src_w2i:
          src_indices.append(self.hparams.unk_id)
          src_unk_count += 1
        else:
          src_indices.append(src_w2i[src_tok])

      trg_w2i = self.trg_w2i_list[i]
      for trg_tok in trg_tokens:
        if trg_tok not in trg_w2i:
          trg_indices.append(self.hparams.unk_id)
          trg_unk_count += 1
        else:
          trg_indices.append(trg_w2i[trg_tok])

      src_indices.append(self.hparams.eos_id)
      trg_indices.append(self.hparams.eos_id)
      src_data.append(src_indices)
      trg_data.append(trg_indices)
      line_count += 1
      if line_count % 10000 == 0:
        print("processed {} lines".format(line_count))
    print("src_unk={}, trg_unk={}".format(src_unk_count, trg_unk_count))
    assert len(src_data) == len(trg_data)
    return src_data, trg_data

  def _build_vocab(self, vocab_file, max_vocab_size=None):
    i2w = []
    w2i = {}
    i = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
      for line in f:
        w = line.strip()
        w2i[w] = i
        i2w.append(w)
        i += 1
        if max_vocab_size and i >= max_vocab_size:
          break
    assert i2w[self.hparams.pad_id] == '<pad>'
    assert i2w[self.hparams.unk_id] == '<unk>'
    assert i2w[self.hparams.bos_id] == '<s>'
    assert i2w[self.hparams.eos_id] == '<\s>'
    assert w2i['<pad>'] == self.hparams.pad_id
    assert w2i['<unk>'] == self.hparams.unk_id
    assert w2i['<s>'] == self.hparams.bos_id
    assert w2i['<\s>'] == self.hparams.eos_id
    return i2w, w2i
