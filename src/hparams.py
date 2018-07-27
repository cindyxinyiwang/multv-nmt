

class HParams(object):
  def __init__(self, **args):
    self.pad = "<pad>"
    self.unk = "<unk>"
    self.bos = "<s>"
    self.eos = "<\s>"
    self.pad_id = 0
    self.unk_id = 1
    self.bos_id = 2
    self.eos_id = 3

    self.batcher = "sent"
    self.batch_size = 32
    self.src_vocab_size = None
    self.trg_vocab_size = None

    self.inf = float("inf")

    for name, value in args.items():
      setattr(self, name, value)
    if hasattr(self, 'train_src_file_list'):
      self.train_src_file_list = self.train_src_file_list.split(',')
    if hasattr(self, 'train_trg_file_list'):
      self.train_trg_file_list = self.train_trg_file_list.split(',')   
    if hasattr(self, 'src_vocab_list'):
      self.src_vocab_list = self.src_vocab_list.split(',')
    if hasattr(self, 'trg_vocab_list'):
      self.trg_vocab_list = self.trg_vocab_list.split(',')