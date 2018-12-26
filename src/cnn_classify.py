import numpy as np
import argparse
import time
import shutil
import gc
import random
import subprocess
import re

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import DataUtil
from mult_data_utils import MultDataUtil
from hparams import *
from utils import *

parser = argparse.ArgumentParser(description="classify")

parser.add_argument("--semb", type=str, default=None, help="[mlp|dot_prod|linear]")
parser.add_argument("--semb_vsize", type=int, default=None, help="how many steps to write log")
parser.add_argument("--trg_no_char", action="store_true", help="load an existing model")
parser.add_argument("--shuffle_train", action="store_true", help="load an existing model")
parser.add_argument("--ordered_char_dict", action="store_true", help="load an existing model")
parser.add_argument("--bpe_ngram", action="store_true", help="bpe ngram")

parser.add_argument("--load_model", action="store_true", help="load an existing model")
parser.add_argument("--reset_output_dir", action="store_true", help="delete output directory if it exists")
parser.add_argument("--output_dir", type=str, default="outputs", help="path to output directory")
parser.add_argument("--log_every", type=int, default=50, help="how many steps to write log")
parser.add_argument("--eval_every", type=int, default=500, help="how many steps to compute valid ppl")
parser.add_argument("--clean_mem_every", type=int, default=10, help="how many steps to clean memory")
parser.add_argument("--eval_bleu", action="store_true", help="if calculate BLEU score for dev set")
parser.add_argument("--beam_size", type=int, default=5, help="beam size for dev BLEU")
parser.add_argument("--poly_norm_m", type=float, default=1, help="beam size for dev BLEU")
parser.add_argument("--ppl_thresh", type=float, default=20, help="beam size for dev BLEU")
parser.add_argument("--max_trans_len", type=int, default=300, help="beam size for dev BLEU")
parser.add_argument("--merge_bpe", action="store_true", help="if calculate BLEU score for dev set")
parser.add_argument("--dev_zero", action="store_true", help="if eval at step 0")

parser.add_argument("--cuda", action="store_true", help="GPU or not")
parser.add_argument("--decode", action="store_true", help="whether to decode only")

parser.add_argument("--max_len", type=int, default=10000, help="maximum len considered on the target side")
parser.add_argument("--n_train_sents", type=int, default=None, help="max number of training sentences to load")

parser.add_argument("--d_word_vec", type=int, default=288, help="size of word and positional embeddings")
parser.add_argument("--d_char_vec", type=int, default=None, help="size of word and positional embeddings")
parser.add_argument("--d_model", type=int, default=288, help="size of hidden states")
parser.add_argument("--d_inner", type=int, default=512, help="hidden dim of position-wise ff")
parser.add_argument("--n_layers", type=int, default=1, help="number of lstm layers")
parser.add_argument("--n_heads", type=int, default=3, help="number of attention heads")
parser.add_argument("--d_k", type=int, default=64, help="size of attention head")
parser.add_argument("--d_v", type=int, default=64, help="size of attention head")
parser.add_argument("--pos_emb_size", type=int, default=None, help="size of trainable pos emb")

parser.add_argument("--data_path", type=str, default=None, help="path to all data")
parser.add_argument("--train_src_file_list", type=str, default=None, help="source train file")
parser.add_argument("--train_trg_file_list", type=str, default=None, help="target train file")
parser.add_argument("--dev_src_file_list", type=str, default=None, help="source valid file")
parser.add_argument("--dev_src_file", type=str, default=None, help="source valid file")
parser.add_argument("--dev_trg_file_list", type=str, default=None, help="target valid file")
parser.add_argument("--dev_trg_file", type=str, default=None, help="target valid file")
parser.add_argument("--dev_ref_file_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--dev_trg_ref", type=str, default=None, help="target valid file for reference")
parser.add_argument("--dev_file_idx_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--src_vocab_list", type=str, default=None, help="source vocab file")
parser.add_argument("--trg_vocab_list", type=str, default=None, help="target vocab file")
parser.add_argument("--test_src_file_list", type=str, default=None, help="source test file")
parser.add_argument("--test_src_file", type=str, default=None, help="source test file")
parser.add_argument("--test_trg_file_list", type=str, default=None, help="target test file")
parser.add_argument("--test_file_idx_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--test_trg_file", type=str, default=None, help="target test file")
parser.add_argument("--src_char_vocab_from", type=str, default=None, help="source char vocab file")
parser.add_argument("--src_char_vocab_size", type=str, default=None, help="source char vocab file")
parser.add_argument("--trg_char_vocab_from", type=str, default=None, help="source char vocab file")
# multi data util options
parser.add_argument("--lang_file", type=str, default=None, help="language code file")
parser.add_argument("--src_vocab", type=str, default=None, help="source vocab file")
parser.add_argument("--src_vocab_from", type=str, default=None, help="list of source vocab file")
parser.add_argument("--trg_vocab", type=str, default=None, help="source vocab file")

parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--valid_batch_size", type=int, default=20, help="batch_size")
parser.add_argument("--batcher", type=str, default="sent", help="sent|word. Batch either by number of words or number of sentences")
parser.add_argument("--n_train_steps", type=int, default=100000, help="n_train_steps")
parser.add_argument("--n_train_epochs", type=int, default=0, help="n_train_epochs")
parser.add_argument("--dropout", type=float, default=0., help="probability of dropping")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_dec", type=float, default=0.5, help="learning rate decay")
parser.add_argument("--n_warm_ups", type=int, default=0, help="lr warm up steps")
parser.add_argument("--lr_schedule", action="store_true", help="whether to use transformer lr schedule")
parser.add_argument("--clip_grad", type=float, default=5., help="gradient clipping")
parser.add_argument("--l2_reg", type=float, default=0., help="L2 regularization")
parser.add_argument("--patience", type=int, default=-1, help="patience")
parser.add_argument("--eval_end_epoch", action="store_true", help="whether to reload the hparams")

parser.add_argument("--seed", type=int, default=19920206, help="random seed")

parser.add_argument("--init_range", type=float, default=0.1, help="L2 init range")
parser.add_argument("--init_type", type=str, default="uniform", help="uniform|xavier_uniform|xavier_normal|kaiming_uniform|kaiming_normal")

args = parser.parse_args()




