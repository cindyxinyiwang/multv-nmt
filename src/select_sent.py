import numpy as np

def get_lan_order(base_lan, lan_dist_file="ted-train-vocab.mtok.sim-ngram.graph"):
  dists = {}
  with open(lan_dist_file, "r") as myfile:
    for line in myfile:
      base, ref, dist = line.split()
      dist = int(dist)
      if base == base_lan:
        dists[ref] = dist
  ordered_lans = sorted(dists.items(), key=lambda kv:kv[1])
  print(ordered_lans)
  #exit(0)
  return ordered_lans

if __name__ == "__main__":
  IL = "bel"
  RL = "rus"
  
  tar_vocab = "data/{}_eng/ted-train.mtok.{}.ochar4vocab".format(IL, IL)
  tar_eng = "data/{}_eng/ted-train.mtok.spm8000.eng".format(RL)
  
  langs = get_lan_order(IL)
  langs = [kv[0] for kv in langs[:-2]][::-1]
  #aze
  #langs = ["por", "ces", "rus"]
  #langs = ["ind", "dan", "epo", "est", "eus", "swe"]
  #bel
  #langs = ["tur", "por", "ces"]
  #langs = ["ukr", "bul", "mkd","kaz","srp"]
  #glg
  #langs = ["tur", "rus", "ces"]
  #langs = ["spa", "ita", "fra", "ron", "epo"]
  #ces
  #langs = ["tur", "por", "rus"]
  #langs = ["slv", "hrv", "srp", "bos"]
  langs_count = [6000, 6000, 6000, 6000]
  data_inputs = []
  data_trgs = []
  data_outputs = []
  data_outtrgs = []
  
  for lan in langs:
    data_inputs.append("data/{}_eng/ted-train.mtok.{}".format(lan, lan))
    data_trgs.append("data/{}_eng/ted-train.mtok.spm8000.eng".format(lan))
    data_outputs.append("data/{}_eng/ted-train.mtok.{}.{}seleng".format(lan, lan, IL))
    data_outtrgs.append("data/{}_eng/ted-train.mtok.spm8000.eng.{}seleng".format(lan, IL))
  
  #for inp, out, trg, outtrg, count in zip(data_inputs, data_outputs, data_trgs, data_outtrgs, langs_count):
  #  inp = np.array(open(inp, 'r').readlines())
  #  out = open(out, 'w')
  #  trg = np.array(open(trg, 'r').readlines())
  #  outtrg = open(outtrg, 'w')
  #
  #  indices = np.random.randint(0, len(inp))
  #  inp = inp[indices]
  #  trg = trg[indices]
  #  for sent, t in zip(inp, trg):
  #    out.write(sent)
  #    outtrg.write(t)
  #  out.close()
  #  outtrg.close()
  
  eng = set([])
  with open(tar_eng, 'r') as myfile:
    for line in myfile:
      eng.add(line)
  for inp, out, trg, outtrg in zip(data_inputs, data_outputs, data_trgs, data_outtrgs):
    inp = open(inp, 'r')
    out = open(out, 'w')
    trg = open(trg, 'r')
    outtrg = open(outtrg, 'w')
    for sent, t in zip(inp, trg):
      if t not in eng:
        out.write(sent)
        outtrg.write(t)
        eng.add(t)
    out.close()
    outtrg.close()
  
  
