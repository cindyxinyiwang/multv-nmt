import numpy as np

def get_lan_order(base_lan):
  lan_dist_file = "ted-train-vocab.mtok.sim-ngram.graph"
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
  return [kv[0] for kv in ordered_lans[:-2]][::-1]

tar_vocab = "data/glg_eng/ted-train.mtok.glg.ochar4vocab"
tar_eng = "data/ces_eng/ted-train.mtok.spm8000.eng"

#aze
#langs = ["por", "ces", "rus"]
#langs = ["ind", "dan", "epo", "est", "eus", "swe"]
langs = get_lan_order("slk")
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
  data_outputs.append("data/{}_eng/ted-train.mtok.{}.slkseleng".format(lan, lan))
  data_outtrgs.append("data/{}_eng/ted-train.mtok.spm8000.eng.slkseleng".format(lan))

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


#vocab = set([])
#with open(tar_vocab, 'r') as myfile:
#  for line in myfile:
#    vocab.add(line.strip())
#
#for inp, out, trg, outtrg in zip(data_inputs, data_outputs, data_trgs, data_outtrgs):
#  inp = open(inp, 'r')
#  out = open(out, 'w')
#  trg = open(trg, 'r')
#  outtrg = open(outtrg, 'w')
#  for sent, t in zip(inp, trg):
#    matched_word = 0
#    words = sent.split()
#    for word in words:
#      max_len = len(word)
#      matched_char = 0
#      for l in range(3, max_len):
#        for i in range(max_len-l):
#          char = word[i:i+l]
#          if char in vocab:
#            matched_char += 1
#      if matched_char: matched_word += 1
#    if matched_word / len(words) > 0.5:
#      out.write(sent)
#      outtrg.write(t)
#  out.close()
#  outtrg.close()
