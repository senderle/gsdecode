#!/usr/local/bin/pypy

import math
import collections
import random
import sys
import re
import string
import argparse
import heapq

def load_text(fn, space_pad=3):
    with open(fn) as datafile:
        text_raw = datafile.read()
        rex = '[^{}]'.format(string.printable)
        text = re.sub(rex, ' ', text_raw)
        return text

def get_ngram_probabilities(text, n):
    if n < 1:
        raise ValueError("Minimum ngram size is 1")

    all_ngrams = []
    for i in xrange(1, n + 1):
        ngrams = count_ngrams(text, i)
        ngrams = categorical_log_probabilities(ngrams)
        for j, sub_ngrams in enumerate(all_ngrams):
            for ng in ngrams:
                ngrams[ng] -= sub_ngrams[ng[:j + 1]]
        all_ngrams.append(ngrams)

    all_ngrams.reverse()
    return all_ngrams

def categorical_log_probabilities(counts):
    log_total = math.log(sum(counts.itervalues()))
    return {key:math.log(counts[key]) - log_total for key in counts}

def count_ngrams(seq, n=3):
    ngrams = (seq[i:i + n] for i in range(len(seq) - n - 1))
    return dict(collections.Counter(ngrams))

class GibbsSampler(object):
    def __init__(self, ngram_probabilities,
                 n_cycles, burn_in, display_interval,
                 jump_probability):

        self.ngram_probabilities = ngram_probabilities
        self.ngram_size = len(ngram_probabilities)

        self.n_cycles = n_cycles
        self.burn_in = burn_in
        self.display_interval = display_interval
        self.jump_probability = jump_probability

        self.coded_string = None

    def gibbs_cycle(self, coded_string):
        self.coded_string = coded_string

        init_ngram_p = self.ngram_probabilities[self.ngram_size - 1]
        key = init_key(init_ngram_p, coded_string)

        samples = collections.Counter()
        max_key = key[:]
        max_prob = self.gibbs_probability(key)

        for i in xrange(self.n_cycles):
            offset = random.randrange(0, len(key))
            for key_ix in xrange(len(key)):
                self.gibbs_sample(key_ix, offset, key)

            prob = self.gibbs_probability(key)
            if prob >= max_prob * 1.5:
                new_prob, new_key = self.maximize_key(key)
                if new_prob > max_prob:
                    max_prob, max_key = new_prob, new_key

            if i % self.display_interval == 0:
                print "Iteration {}:".format(i)
                self.display_key("Current key", key)
                self.display_key("Most probable key", max_key)

    def display_key(self, label, key):
        coded_string = self.coded_string

        prob = self.gibbs_probability(key)
        prob_per_char = prob / len(coded_string)

        print '{}: '.format(label)
        print
        print [key[x] for x in xrange(len(key))]
        print
        print '{} output: '.format(label)
        print
        print ''.join(key[s] for s in coded_string)
        print
        print '{} log probability, total and per character: '.format(label)
        print
        print prob, prob_per_char
        print

    def key_from_samples(self, samples, key):
        top = samples.most_common()
        sample_key = {}
        symbols = set(xrange(len(key)))
        chars = set(key)

        for (symbol, char), count in top:
            if symbol in symbols and char in chars:
                sample_key[symbol] = char
                symbols.remove(symbol)
                chars.remove(char)

        for symbol in xrange(len(key)):
            if symbol not in sample_key:
                sample_key[symbol] = 'X'

        return [sample_key[x] for x in xrange(len(sample_key))]

    def maximize_key(self, key, max_prob=None):
        prob = self.gibbs_probability(key)
        if max_prob is None:
            max_prob = prob - 1

        key = key[:]
        while prob > max_prob:
            max_prob = prob
            offset = random.randrange(0, len(key))
            for key_ix in xrange(len(key)):
                self.hill_step(key_ix, offset, key)
            prob = self.gibbs_probability(key)
            self.display_key("Hill climbing key", key)

        return max_prob, key

    def gibbs_sample(self, start, offset, key):
        # For every index in key following `start`, swap index with current
        # index and recalculate probability. (A modified fisher-yates shuffle.)
        # Accepts an offset parameter to cut the "deck" before shuffling.

        start_ix = (start + offset) % len(key)
        if random.random() < self.jump_probability:
            swap_ix = random.randrange(start, len(key))
            swap_ix = (swap_ix + offset) % len(key)
        else:
            probs = []
            new_key = []
            for ix in xrange(start, len(key)):
                ix = (ix + offset) % len(key)
                new_key[:] = key
                new_key[ix], new_key[start_ix] = new_key[start_ix], new_key[ix]
                p = self.gibbs_probability(new_key)
                probs.append(p)

            swap_ix = categorical_sample(probs)
            swap_ix = (swap_ix + start_ix) % len(key)

        key[start_ix], key[swap_ix] = key[swap_ix], key[start_ix]

    def hill_step(self, start, offset, key):
        # Same as above, but no shuffling; this always takes the maximum.
        # Accepts an offset parameter to cut the "deck" before shuffling.

        start_ix = (start + offset) % len(key)
        probs = []
        new_key = []
        for ix in xrange(start, len(key)):
            ix = (ix + offset) % len(key)
            new_key[:] = key
            new_key[ix], new_key[start_ix] = new_key[start_ix], new_key[ix]
            p = self.gibbs_probability(new_key)
            probs.append(p)

        swap_ix = max(range(len(probs)), key=probs.__getitem__)
        swap_ix = (swap_ix + start_ix) % len(key)

        key[start_ix], key[swap_ix] = key[swap_ix], key[start_ix]

    def gibbs_probability_5grams(self, key):
        # A fast 5-gram benchmark for speeding up the
        # gibbs probabilitiy code below. The generic gibbs is
        # currently about 90% as fast as this on specialized gibbs
        # on pypy -- much slower using vanilla python or cython though.
        quintgram_p, quadgram_p, trigram_p, bigram_p, unigram_p = \
            self.ngram_probabilities

        decoded_string = ''.join(key[c] for c in self.coded_string)
        decoded_ngrams = [decoded_string[i:i + 5] for i in
                          xrange(0, len(decoded_string) - 4)]
        p = 0
        for decoded_ng in decoded_ngrams:
            if decoded_ng in quintgram_p:
                p += quintgram_p[decoded_ng]
            elif decoded_ng[1:] in quadgram_p:
                p += quadgram_p[decoded_ng[1:]] * 5.0 / 4
            elif decoded_ng[2:] in trigram_p:
                p += trigram_p[decoded_ng[2:]] * 5.0 / 3
            elif decoded_ng[3:] in bigram_p:
                p += bigram_p[decoded_ng[3:]] * 5.0 / 2
            else:
                p += unigram_p[decoded_ng[4:]] * 5.0
        return p

    def gibbs_probability(self, key):
        ngram_p = self.ngram_probabilities
        ngram_max = self.ngram_size
        decoded_string = ''.join([key[c] for c in self.coded_string])

        p = 0
        string_end = len(decoded_string)
        for end in xrange(ngram_max, string_end):
            start = end - ngram_max
            for ngp in ngram_p:
                ngram = decoded_string[start:end]
                if ngram in ngp:
                    ngram_size = end - start
                    p += ngram_max * ngp[ngram] / (ngram_size)
                    break
                start += 1
        return p

# Until I can make this a _pure cython_ function, there's no
# point in compiling this. It's faster to just run the above
# through pypy.

    #def gibbs_probability_ngrams(self, key):
    #    ngram_p = self.ngram_probabilities
    #    decoded_string = ''.join([key[c] for c in self.coded_string])
    #
    #    cdef int ngram_max, string_end, start, end
    #    cdef double p
    #    ngram_max = self.ngram_size

    #    p = 0
    #    string_end = len(decoded_string)
    #    for end in xrange(ngram_max, string_end):
    #        start = end - ngram_max
    #        for ngp in ngram_p:
    #            ngram = decoded_string[start:end]
    #            if ngram in ngp:
    #                ngram_size = end - start
    #                p += ngram_max * ngp[ngram] / (ngram_size)
    #                break
    #            start += 1
    #    return p

def categorical_sample(probs):
    maxp = max(probs)
    probs = [math.exp(p - maxp) for p in probs]
    total = sum(probs)
    if total == 0:
        probs = [1] * len(probs)
        total = sum(probs)
    r = random.random()
    acc = 0
    for i, p in enumerate(probs):
        acc += float(p) / total
        if r <= acc:
            return i

def init_key(model_ngrams, encoded_mystery):
    n = len(next(model_ngrams.iterkeys()))
    mystery_ngrams = count_ngrams(tuple(encoded_mystery), n)

    ordered_model = sorted(model_ngrams,
                           key=model_ngrams.__getitem__,
                           reverse=True)
    ordered_mystery = sorted(mystery_ngrams,
                             key=mystery_ngrams.__getitem__,
                             reverse=True)

    alphabet = set(c for ng in model_ngrams for c in ng)
    indices = set(xrange(len(alphabet)))

    key = [None] * len(alphabet)
    assigned = set()
    for model_ng, mystery_ng in zip(ordered_model, ordered_mystery):
        for ix, char in zip(mystery_ng, model_ng):
            if ix in assigned or char in assigned:
                continue
            key[ix] = char
            assigned.add(ix)
            assigned.add(char)

    open_indices = indices - assigned
    unassigned_chars = alphabet - assigned
    for ix, char in zip(open_indices, unassigned_chars):
        key[ix] = char

    return key

def parse_args():
    gsd_parser = argparse.ArgumentParser(description='A substitution cipher '
        'decoder that uses a modified version of Gibbs Sampling that may '
        'actually be closer to metropolis-hastings sampling, because of '
        'the way it uses heuristics to escape local minima.')
    gsd_parser.add_argument('-c', '--num-cycles', type=int, default=100,
        metavar='number', help='The total number of gibbs sampling cycles to '
        'complete.')
    gsd_parser.add_argument('-b', '--burn-in', type=int, default=20,
        metavar='number', help='The number of samples to discard before '
        'accumulating samples. Defaults to 20.')
    gsd_parser.add_argument('-d', '--display-interval', type=int, default=1,
        metavar='number', help='The frequency with which to display current '
        'key information.')
    gsd_parser.add_argument('-j', '--jump-probability', type=float,
        default=0.025, metavar='0.0-1.0', help='The probability of jumping '
        'from one state to another randomly. This is a heuristic for '
        'escaping local minima.')
    gsd_parser.add_argument('-n', '--ngram-size', type=int, default=5,
        metavar='number', choices=[2, 3, 4, 5, 6],
        help='The number of characters to use for the ngram model. Defaults '
        'to 5. Models larger than 6 tend to underperform.')
    gsd_parser.add_argument('-m', '--model-file', type=str,
        metavar='filename', action='append', help='The name of a file with '
        'which to analyize character n-gram frequencies. May be specified '
        'multiple times, and must be specified at least once.')
    gsd_parser.add_argument('mystery', type=str, metavar='filename',
        help='The name of a file containing the text to be encrypted '
        'and decrypted.')
    return gsd_parser.parse_args()

def main(argv):

    args = parse_args()

    model_texts = [load_text(fn) for fn in args.model_file]
    model_text = ' '.join(model_texts)
    ngram_probabilities = get_ngram_probabilities(model_text, args.ngram_size)

    chars = list(set(model_text))
    random.shuffle(chars)
    code = {c:i for i, c in enumerate(chars)}

    mystery = load_text(args.mystery)

    encoded_mystery = [code[c] for c in mystery]

    print "Code:",
    print [c for i, c in sorted((i, c) for c, i in code.iteritems())]

    # Decide on a sample length based on the number of possible
    # characters...  should be something like 2x or 3x smallest
    # sample, then that value ** 2, ** 3, and so on. ?
    #mystery_len = len(mystery)
    #mystery_loglen = math.log(mystery_len, 10)
    #mystery_chunks = [

    decoder = GibbsSampler(ngram_probabilities,
                           args.num_cycles, args.burn_in,
                           args.display_interval,
                           args.jump_probability)
    decoder.gibbs_cycle(encoded_mystery)

if __name__ == '__main__':
    main(sys.argv)  # this makes it easier to optionally compile with Cython
                    # for a modest speedup