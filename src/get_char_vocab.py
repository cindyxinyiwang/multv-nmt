import sys

n = int(sys.argv[1])
vocab = {}
for line in sys.stdin:
  toks = line.split()
  for w in toks:
    for i in range(len(w)):
      for j in range(i+1, min(i+n, len(w))+1):
        char = w[i:j]
        if char not in vocab:
          vocab[char] = 0
        vocab[char] += 1

vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)
print("<pad>")
print("<unk>")
print("<s>")
print("<\s>")

for w, c in vocab:
  print(w)
