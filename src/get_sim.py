
vocab_size = 8000
base_lan = "aze"
#lan_lists = ["rus", "por", "ces"]
lan_lists = ["aze", "tur", "rus", "por", "ces"]


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

if __name__ == "__main__":
  sim_by_ngram(base_lan, lan_lists)
