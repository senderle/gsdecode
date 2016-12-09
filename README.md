# gsdecode
### A toy implementation of a substitution cipher solver using Gibbs(ish) sampling.

This is a Gibbs sampler over substitution ciphers. Each cipher is represented as a permutation of the available characters, and the sampler walks over the space of character permutations.

Practically speaking, this is useless, since nobody uses substitution ciphers for serious encryption. But it provides an interesting starting point for thinking about how to discern likely words from unlikely words based on known character n-gram distributions.

### Algorithm details

The goodness-of-fit for each cipher is calculated using the naive Bayes log probabiliy of the n-grams generated when the cipher is used to decode a given ciphertext. For missing n-grams, it steps downward, substituting the probability for the required `n-1`-gram. It continues stepping down until `n` is 1.

Based on the evaluations that result, it picks a pair of characters in the cipher to swap, producing a new permutation. Character swaps that appear to produce better results are more likely to be chosen, but all possible swaps are considered and assigned a non-zero probability.

This doesn't appear to represent a well-formed probability distribution -- not surprisingly! -- so it doesn't behave as nicely as a Gibbs Sampler based on a legitimate probabilistic derivation. But I'm not equipped to produce such a derivation, so instead, this use some tricks to improve performance. 

1. It hill-climbs from time to time, taking the _best_ result every time, instead of chosing a result using fair sampling. This speeds up convergence to reasonable results, and can quickly decrypt longer ciphertexts. But it also tends to get stuck in bad local optima.

2. To get out of bad local optima, this occasionally takes a completely random result. That can have positive or negative results, depending on whether the local optimum is a good one or not, but on average, it ensures that most plausible areas of the cipher permutation space are explored.

This script runs very slowly on standard Python. I recommend running it with Pypy instead.

To see a simple example of its output, run

    pypy gsdecode.py -m models/rasselas.ascii.upper.txt -d 10 -c 100 mysteries/mystery2.txt

This works with texts in any language, assuming you have a large enough model text (for calculating n-gram probabilities) and know the probable language of the ciphertext. `mysteries` contains a range of different plain texts, which are encrypted using a randomly-chosen cipher before running the sampler. (To use this on "real" encrypted texts you'd need to tweak the script, but that's OK, because nobody seriously uses substitution ciphers for encryption anyway. This is just for fun!)

The `mysteries` directory contains several hard problems, including a few English lipograms and other phrases with odd distributional propeties. These really challenge the engine. It mostly manages to get close to the correct solutions, but it desperately wants `e` to appear, and often decides that newline characters are actually upper- or lowercase `e`s.

### Output Example

```
$ cat mysteries/mystery5.txt 
I shoot the hippopotamous with bullets made of platinum
because if I use leaden ones his hide is sure to flatten em.

$ pypy gsdecode.py -m models/rasselas.ascii.upper.txt -d 10 -c 100 mysteries/mystery5.txt 
```
[... Many lines of output here... ]

Final output:

```
Hill climbing key: 

['w', 'o', "'", 'h', 'X', ';', 'J', '!', 'P', 'r', 'p', 'u', '"', 
 'A', 'f', 'q', 'n', 'v', 'Y', 'L', 'V', 'D', 'z', 'Z', 'H', '\n', 
 's', 'a', 'b', 'l', ',', 'G', 'K', 'C', 'O', 'M', 'N', 'W', 'x', 
 '_', 'S', 'Q', 'y', ' ', 'g', 'R', 'j', ')', 'T', 'U', 'e', 'E', 
 'I', 'm', '.', 'k', 'c', '-', 'B', 'i', 'd', ':', '?', 'F', '(', 
 't']

Hill climbing key output: 

y shoot the hippopotamous with rullets made of platinum
recause if y use leaden ones his hide is sube to flatten em,


Hill climbing key log probability, total and per character: 

-274.614803669 -2.34713507409
```
