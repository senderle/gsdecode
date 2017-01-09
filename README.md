# gsdecode
### A toy implementation of a substitution cipher solver using Gibbs(ish) sampling.

This is a Gibbs sampler over substitution ciphers. Each cipher is represented as a permutation of the available characters, and the sampler walks over the space of character permutations.

Practically speaking, this is useless, since nobody uses substitution ciphers for serious encryption. But it provides an interesting starting point for thinking about how to discern likely words from unlikely words based on known character n-gram distributions. I'm thinking about adapting portions of it for de-noising OCR. It wouldn't "normalize," but it might be good at stripping out irredeemably mangled words and pure line noise, while tolerating variant spellings and more isolated OCR errors. More intelligent normalizers could quickly process the rest.

### Algorithm details

The goodness-of-fit for each cipher is calculated using the naive Bayes log probability of the n-grams generated when the cipher is used to decode a given ciphertext. For missing n-grams, it steps downward, substituting the probability for the required `n-1`-gram. It continues stepping down until `n` is 1.

Based on the evaluations observed for each possible pair, it picks one pair in the cipher to swap, producing a new permutation. Character swaps that appear to produce better results are more likely to be chosen, but all possible swaps are considered and assigned a non-zero probability.

This doesn't appear to represent a well-formed probability distribution -- not surprisingly! -- so it doesn't behave as nicely as a Gibbs Sampler based on a legitimate probabilistic derivation. But I'm not equipped to produce such a derivation, so instead, this uses some tricks to improve performance:

1. It hill-climbs from time to time, taking the _best_ result every time, instead of choosing a result using fair sampling. This speeds up convergence to a reasonable area of cipher permutation space, and can quickly decrypt longer ciphertexts. But it also tends to get stuck in bad local optima.

2. To get out of bad local optima, this occasionally chooses a completely random pair to swap. That can have positive or negative results, depending on whether the local optimum is a good one or not, but on average, it ensures that most plausible areas of the cipher permutation space are explored.

This script runs very slowly on standard Python. I recommend running it with Pypy instead.

To see a simple example of its output, run

    pypy gsdecode.py -m models/rasselas.ascii.upper.txt -d 10 -c 100 mysteries/mystery2.txt

This works with texts in any language, assuming you have a large enough model text (for calculating n-gram probabilities) and know the probable language of the ciphertext. `mysteries` contains a range of different plain texts, which are encrypted using a randomly-chosen cipher before running the sampler. (To use this on "real" encrypted texts you'd need to tweak the script, but that's OK, because nobody seriously uses substitution ciphers for encryption anyway. This is just for fun!)

The `mysteries` directory contains several hard problems, including a few English lipograms and other phrases with odd distributional properties. These really challenge the engine. It mostly manages to get close to the correct solutions, but it desperately wants `e` to appear, and often decides that newline characters are actually upper- or lowercase `e`s.

### Output Example

```
$ cat mysteries/mystery5.txt 
I shoot the hippopotamous with bullets made of platinum
because if I use leaden ones his hide is sure to flatten em.

$ pypy3 gsdecode.py decode -m models/middlemarch.txt -d 10 -c 100 mysteries/mystery5.txt
```
[... Many lines of output here... ]

Final output:

```
******************
** Final result **
******************

Most probable key: 

['q', ' ', '0', 'B', 'c', 'v', 'f', 'Z', 'G', 'X', '&', '8', '!', 'V', 'r', 't', 
 's', 'K', 'a', 'D', ']', 'm', ')', 'k', 'J', 'N', '5', '1', 'p', 'F', '_', 'U', 
 'Y', 'j', 'M', '9', '.', 'L', 'i', 'R', 'C', 'n', '2', '6', 'y', 'u', 'g', ',', 
 'W', '?', '\n', 'H', 'Q', 'l', 'w', '(', 'h', 'o', 'S', 'A', ';', "'", '3', 'd', 
 'e', 'I', 'O', 'P', 'x', '-', 'T', ':', '7', 'E', 'z', '[', '"', 'b']

Most probable key output: 

I shoot the hippopotamous with bullets made of platinum
because if I use leaden ones his hide is sure to flatten em.


Most probable key log probability, total and per character: 

-256.60309431622267, -2.1931888403095954
```
