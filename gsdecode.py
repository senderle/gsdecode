#!/usr/local/bin/pypy3

import math
import collections
from collections import abc
import random
import sys
import re
import string
import argparse
import textwrap
import os
import gzip
import zlib
import json

def load_text(filename):
    return list(iter_text(filename))

def iter_text(filename):
    if os.path.isdir(filename):
        files = [os.path.join(filename, f) for f in os.listdir(filename)]
        files = [f for f in files
                 if f.endswith('.txt') and not os.path.isdir(f)]
    else:
        files = [filename]

    for fn in files:
        yield read_text(fn)

def read_text(fn):
    with open(fn) as datafile:
        text_raw = datafile.read()
        rex = '[^{}]'.format(string.printable)  # TODO: Make this unicode-
        text = re.sub(rex, ' ', text_raw)       # friendly.
    return text

def iter_paths(paths):
    for path in paths:
        if os.path.isdir(path):
            files = (os.path.join(path, f) for f in os.listdir(path)
                     if not os.path.isdir(f))
        else:
            files = (path,)

        for f in files:
            yield f

def iter_models(paths, ngram_size=None, uniform_ws=False, lower=False,
                remove_punct=False, remove_digits=False):
    texts = []

    # Find all the compiled models and yield them first,
    # saving the names of plain text files.
    for f in iter_paths(paths):
        if f.endswith('.tgz') or f.endswith('.gz') or f.endswith('tar.gz'):
            try:
                yield NgramModel.from_saved(f)
            except (zlib.error, json.JSONDecodeError):
                continue
        elif f.endswith('.txt'):
            texts.append(f)

    # Then join all the text files together at the end in one big model.
    model_text = ' '.join(read_text(f) for f in texts)
    if not model_text:
        return

    if uniform_ws:
        model_text = uniform_whitespace(model_text)
    if lower:
        model_text = model_text.lower()
    if remove_punct:
        model_text = strip_punctuation(model_text)
    if remove_digits:
        model_text = strip_digits(model_text)
    yield NgramModel(model_text, ngram_size)

def uniform_whitespace(text, _space_rex=re.compile('\s+')):
    return _space_rex.sub(' ', text)

def maketrans_strip(chars):
    return chars.maketrans(chars, ' ' * len(chars))

def strip_punctuation(text, _punct_trans=maketrans_strip(string.punctuation)):
    return text.translate(_punct_trans)

def strip_digits(text, _digits_trans=maketrans_strip('0123456789')):
    return text.translate(_digits_trans)

def get_ngram_probabilities(text, n):
    if n < 1:
        raise ValueError("Minimum ngram size is 1")
    counts = [count_ngrams(text, i) for i in range(n, 0, -1)]
    return counts_to_conditionals(counts)

def counts_to_conditionals(counts):
    """Calculate the probability of the given ngram as the probability
    of the last character appearing given the previous characters.
    This way, we can substitute lower n-gram stats for missing higher
    n-gram stats in an internally consistent way.

    We assume the input counts are ordered longest-first. That is,
    5-grams, then 4-grams, then 3-grams... etc.

    Our approach uses the below version of Bayes' law, where
    p(ng | condition[ng]) is the conditional probability of the ngram
    given its condition, and p(ng, condition[ng]) is the joint
    probability of the ngram and its condition (i.e. the unconditional
    probability of the 5-gram). Here, the "condition" of an n-gram is
    its first (n-1) characters:

        p(ng | condition[ng]) = p(ng, condition[ng]) / p(condition[ng])

    Converted into log-space, we have:

        logp(ng | condition[ng]) =
            logp(ng, condition[ng]) - logp(condition[ng])

    which is easily implemented as an in-place subtraction.
    """

    joints = [categorical_log_probabilities(c) for c in counts]
    conditionals = []
    for joint, condition in zip(joints, joints[1:]):
        for ng in joint:
            joint[ng] -= condition[ng[:-1]]
        conditionals.append(joint)
    conditionals.append(joints[-1])

    return conditionals

# Reverse the process described above. Again, we assume the conditional
# probabilities are ordered longest first, but here we need them
# in the opposite order, so we reverse the order of the input, and
# then reverse the resulting output again.
def conditionals_to_counts(conditionals, text_len):
    conditionals = [c.copy() for c in conditionals]
    conditionals.reverse()
    joints = [conditionals[0]]
    for conditional, condition in zip(conditionals[1:], conditionals):
        for ng in conditional:
            conditional[ng] += condition[ng[:-1]]
        joints.append(conditional)

    counts = [categoricals_to_counts(jt, text_len - n)
              for n, jt in enumerate(joints, start=1)]
    counts.reverse()
    return counts

def categorical_log_probabilities(counts):
    """In log-space, we can treat multiplication as addition and division
    as subtraction. That makes calculations on probabilities much less
    prone to numerical error.

    This is equivalent to dividing the given ngram count by the total
    number of ngrams.
    """
    log_total = math.log(sum(counts.values()))
    result = {key: math.log(counts[key]) - log_total for key in counts}
    return result

def categoricals_to_counts(categoricals, total):
    """Reverse the above process. Because the calculations are all happening
    in log space, there will be low numerical error, and rounding to the
    nearest integer will almost certainly restore the original value.
    """
    log_total = math.log(total)
    return {key: round(math.exp(categoricals[key] + log_total))
            for key in categoricals}

def join_counts(*counts):
    keys = set(k for count in counts for k in count.keys())
    return {key: sum(count[key] if key in count else 0 for count in counts)
            for key in keys}

def count_ngrams(seq, n=3):
    # Count the ngrams generated by moving an n-character window over the text.
    ngrams = (seq[i:i + n] for i in range(len(seq) - n + 1))
    return dict(collections.Counter(ngrams))

class NgramModel(object):
    def __init__(self, text, n=None, lower=True,
                 remove_punctuation=True, remove_digits=True):
        if n is None:
            n = 5

        if text:
            self.n = n
            self.length = len(text)
            self.ngram_probabilities = get_ngram_probabilities(text, n)
        else:
            self.n = 0
            self.length = 0
            self.ngram_probabilities = []

        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits

    @classmethod
    def from_saved(cls, path):
        instance = cls('')
        with gzip.open(path, 'rt') as infile:
            instance.__dict__.update(json.load(infile))
        return instance

    @classmethod
    def from_counts(cls, counts, lower=True, remove_punctuation=True,
                    remove_digits=True):
        instance = cls('',
                       lower=lower,
                       remove_punctuation=remove_punctuation,
                       remove_digits=True)
        instance.n = len(counts)
        instance.length = sum(counts[-1].values())
        instance.ngram_probabilities = counts_to_conditionals(counts)
        return instance

    @classmethod
    def join_all(cls, models):
        models = list(models)
        if len(models) == 1:
            return models[0]

        assert len(set(m.n for m in models)) == 1, (
            'NgramModel.join_all requires that all '
            'models have the same ngram size'
        )

        counts = [m.to_counts() for m in models]
        counts = [join_counts(*n_counts) for n_counts in zip(*counts)]

        lower = any(m.lower for m in models)
        remove_punctuation = any(m.remove_punctuation for m in models)
        remove_digits = any(m.remove_digits for m in models)
        return cls.from_counts(counts,
                               lower,
                               remove_punctuation,
                               remove_digits)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def save(self, path):
        if not path.endswith('.gz'):
            path += '.json.gz'
        with gzip.open(path, 'wt') as out:
            json.dump(self.__dict__, out)

    def join(self, other):
        assert self.n == other.n
        new_counts = zip(self.to_counts(), other.to_counts())
        new_counts = [join_counts(s, o) for s, o in new_counts]
        return NgramModel.from_counts(
            new_counts,
            self.lower or other.lower,
            self.remove_punctuation or other.remove_punctuation,
            self.remove_digits or other.remove_digits
        )

    def to_counts(self):
        return conditionals_to_counts(self.ngram_probabilities, self.length)

    def _process_input(self, string):
        if self.lower:
            string = string.lower()
        if self.remove_punctuation:
            string = strip_punctuation(string)
        if self.remove_digits:
            string = strip_digits(string)
        return string.strip()

    def joint_probability(self, string):
        string = self._process_input(string)
        ngram_p = 0
        for end in range(1, len(string) + 1):
            start = max(0, end - self.n)
            ngram_p += self._lookup_ngram(string[start:end])
        return ngram_p

    def stream_probability(self, string):
        string = self._process_input(string)
        for end in range(1, len(string) + 1):
            start = max(0, end - self.n)
            char_p = self._lookup_ngram(string[start:end])
            yield string[end - 1], char_p

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.joint_probability(key)
        elif isinstance(key, abc.Sequence) and len(key) == 2:
            target_key, given_key = key
            joint_key = target_key + given_key
            joint = self.joint_probability(joint_key)

            given_key = self._process_input(given_key)
            return joint - self._lookup_ngram(given_key)
        else:
            raise ValueError('Invalid NgramModel key: {}'.format(key))

    def _lookup_ngram(self, init_ngram):
        return lookup_ngram(init_ngram, self.ngram_probabilities)

class NgramModelPairwiseClassifier(object):
    def __init__(self, model_zero, model_one, prior=0.5):
        self.prior = prior
        self.model_zero = model_zero
        self.model_one = model_one
        self.model_joint = model_zero.join(model_one)

    def classify_logprob(self, text, prior=None):
        """Bayes law in log-space:

            logp(c | t) = logp(t | c) + log(c) - logp(t)
        """
        prior = math.log(prior) if prior is not None else math.log(self.prior)
        return self.model_one[text] + prior - self.model_joint[text]

    def classify_prob(self, text, prior=None):
        return math.exp(self.classify_logprob(text, prior))

    def classify(self, text, prior=None):
        return self.classify_prob(text, prior) > 0.5

def lookup_ngram_linear_smoothing(init_ngram, ngram_probabilities):
    """Determine the conditional probability of the last character
    in `init_ngram` given the preceding sequence of characters. This
    value is saved in `ngram_probabilities` for ngrams of all lengths
    up to `ngram_max` for ngrams that have been seen before. The
    challenge comes when we try to assign a probability to an unseen
    ngram. Here, we seek the first known lower-order ngram. Suppose
    `init_ngram` is an unseen 5-gram; then we seek the 4-gram ending
    with the same character, dropping the 5-gram's first character.
    If we don't find the 4-gram, we seek a 3-gram, and so on. Then
    we add a penalty term, since the unseen ngram is probably not
    part of any known word. This version of the function applies
    a log-linear penalty that scales as a multiple of the number
    of characters dropped.
    """
    if not init_ngram:
        return 0

    # The log probability of a sample from a uniform
    # distribution over n unigrams (where `n` is the number
    # of possible characters, here the length of the model's
    # unigram distribution.) We increase it by a factor of 1.5
    # since this is a penalty term for unseen characters, which
    # are more likely to be wrong.
    uniform = -math.log(len(ngram_probabilities[-1])) * 2
    ngram_max = len(init_ngram)

    start = 0
    for ngram_table in ngram_probabilities[-ngram_max:]:
        ngram = init_ngram[start:]
        if ngram in ngram_table:
            # p * (1 / len(alphabet)) ** n_chars_dropped
            return ngram_table[ngram] + uniform * start
        else:
            start += 1

    # If we get to this point, then the unigram itself is
    # unkown. We apply a strong penalty.
    return uniform * ngram_max

def lookup_ngram_geometric_smoothing(init_ngram, ngram_probabilities):
    """See above. This version of the function applies a penalty
    that scales as a geometric function of the number of characters
    dropped.
    """
    if not init_ngram:
        return 0

    ngram_max = len(init_ngram)

    start = 0
    for ngram_table in ngram_probabilities[-ngram_max:]:
        ngram = init_ngram[start:]
        if ngram in ngram_table:
            # p ** (len(init_ngram) / (len(init_ngram) - n_chars_dropped))
            return ngram_max * ngram_table[ngram] / len(ngram)
        else:
            start += 1

    # If we get to this point, then the unigram itself is
    # unkown. We apply a strong penalty.
    return -math.log(len(ngram_probabilities[-1])) * ngram_max * 2

lookup_ngram = lookup_ngram_geometric_smoothing

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

        max_key = key[:]
        max_prob = self.gibbs_probability(key)

        # The main loop. We cycle many times over two steps...
        for i in range(self.n_cycles):
            # First, we "shuffle" a pair of cipher characters, but in a way
            # that's weighted toward better results.
            offset = random.randrange(0, len(key))
            for key_ix in range(len(key)):
                self.gibbs_sample(key_ix, offset, key)

            # Second, we check to see if we've made a breakthrough. If so,
            # we see how much more ground we can claim quickly and greedily.
            prob = self.gibbs_probability(key)
            if prob >= max_prob * 1.5:
                new_prob, new_key = self.maximize_key(key)
                if new_prob > max_prob:
                    max_prob, max_key = new_prob, new_key

            if i % self.display_interval == 0:
                print("Iteration {}:".format(i))
                self.display_key("Current key", key)
                self.display_key("Most probable key", max_key)

        print('******************')
        print('** Final result **')
        print('******************')
        print()
        self.display_key("Most probable key", max_key)

    def display_key(self, label, key):
        coded_string = self.coded_string

        prob = self.gibbs_probability(key)
        prob_per_char = prob / len(coded_string)

        print('{}: '.format(label))
        print()
        print([key[x] for x in range(len(key))])
        print()
        print('{} output: '.format(label))
        print()
        print(''.join(key[s] for s in coded_string))
        print()
        print('{} log probability, total and per character: '.format(label))
        print()
        print('{}, {}'.format(prob, prob_per_char))
        print()

    def key_from_samples(self, samples, key):
        # This picks a key given a colleciton of samples. But hill-climbing
        # works better, so this code isn't used anymore.
        top = samples.most_common()
        sample_key = {}
        symbols = set(range(len(key)))
        chars = set(key)

        for (symbol, char), count in top:
            if symbol in symbols and char in chars:
                sample_key[symbol] = char
                symbols.remove(symbol)
                chars.remove(char)

        for symbol in range(len(key)):
            if symbol not in sample_key:
                sample_key[symbol] = 'X'

        return [sample_key[x] for x in range(len(sample_key))]

    def maximize_key(self, key, max_prob=None):
        prob = self.gibbs_probability(key)
        if max_prob is None:
            max_prob = prob - 1

        key = key[:]
        i = 0
        while prob > max_prob:
            i += 1
            max_prob = prob
            offset = random.randrange(0, len(key))
            for key_ix in range(len(key)):
                self.hill_step(key_ix, offset, key)
            prob = self.gibbs_probability(key)

        return max_prob, key

    def gibbs_sample(self, start, offset, key):
        # For every index in key following `start`, swap index with current
        # index and recalculate probability. Then pick an index pair to swap,
        # based on the calculated probabilities. Occasionally, this totally
        # ignores reason and picks a completely random pair to swap.
        # (This behavior is goverend by `self.jump_probability`).

        # Accepts an offset parameter to cut the "deck" before shuffling; this
        # seems to reduce the amount of time spent in bad local optima.

        start_ix = (start + offset) % len(key)
        if random.random() < self.jump_probability:
            swap_ix = random.randrange(start, len(key))
            swap_ix = (swap_ix + offset) % len(key)
        else:
            probs = []
            new_key = []
            for ix in range(start, len(key)):
                ix = (ix + offset) % len(key)
                new_key[:] = key
                new_key[ix], new_key[start_ix] = new_key[start_ix], new_key[ix]
                p = self.gibbs_probability(new_key)
                probs.append(p)

            swap_ix = categorical_sample(probs)
            swap_ix = (swap_ix + start_ix) % len(key)

        key[start_ix], key[swap_ix] = key[swap_ix], key[start_ix]

    def hill_step(self, start, offset, key):
        # Same as above, but no randomization; this always takes the maximum.
        # Accepts an offset parameter to cut the "deck" before shuffling.

        start_ix = (start + offset) % len(key)
        probs = []
        new_key = []
        for ix in range(start, len(key)):
            ix = (ix + offset) % len(key)
            new_key[:] = key
            new_key[ix], new_key[start_ix] = new_key[start_ix], new_key[ix]
            p = self.gibbs_probability(new_key)
            probs.append(p)

        swap_ix = max(range(len(probs)), key=probs.__getitem__)
        swap_ix = (swap_ix + start_ix) % len(key)

        key[start_ix], key[swap_ix] = key[swap_ix], key[start_ix]

    # A slightly more correct (but possibly slower)
    # calculation than the ones below.
    def gibbs_probability(self, key):
        ngram_p = self.ngram_probabilities
        ngram_max = self.ngram_size
        decoded_string = ''.join([key[c] for c in self.coded_string])

        p = 0
        string_end = len(decoded_string)
        for end in range(1, string_end + 1):
            start = max(0, end - ngram_max)
            p += lookup_ngram_geometric_smoothing(decoded_string[start:end],
                                                  ngram_p)
        return p

    def gibbs_probability_5grams(self, key):
        # A fast 5-gram benchmark for speeding up the gibbs probabilitiy
        # code below. The generic gibbs (below) is currently about 90%
        # as fast as this on pypy -- much slower using vanilla python
        # or cython though.
        quintgram_p, quadgram_p, trigram_p, bigram_p, unigram_p = \
            self.ngram_probabilities

        decoded_string = ''.join(key[c] for c in self.coded_string)
        decoded_ngrams = [decoded_string[i:i + 5] for i in
                          range(0, len(decoded_string) - 4)]
        p = 0

        # Next character probabilites. Here, recall that log probabilities
        # are negative so additions and multiplications make values
        # smaller. The multipliers here ensure that when we have only
        # low-n-gram information available, we don't over-weight that
        # information. More intelligent approaches might substitute
        # probabilities based on character class-grams; for example,
        # we might reasonably guess that a character following three
        # vowels is highly unlikely to be a vowel itself.
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

    def gibbs_probability_original(self, key):
        ngram_p = self.ngram_probabilities
        ngram_max = self.ngram_size
        decoded_string = ''.join([key[c] for c in self.coded_string])

        p = 0
        string_end = len(decoded_string)
        for end in range(ngram_max, string_end):
            start = end - ngram_max
            for ngp in ngram_p:
                ngram = decoded_string[start:end]
                if ngram in ngp:
                    ngram_size = end - start
                    p += ngram_max * ngp[ngram] / (ngram_size)
                    break
                start += 1
        return p

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
    n = len(next(iter(model_ngrams.keys())))
    mystery_ngrams = count_ngrams(tuple(encoded_mystery), n)

    ordered_model = sorted(model_ngrams,
                           key=model_ngrams.__getitem__,
                           reverse=True)
    ordered_mystery = sorted(mystery_ngrams,
                             key=mystery_ngrams.__getitem__,
                             reverse=True)

    alphabet = set(c for ng in model_ngrams for c in ng)
    indices = set(range(len(alphabet)))

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

def decode(args):
    models = iter_models(args.model_file,
                         args.ngram_size,
                         args.lower,
                         args.remove_punctuation,
                         args.remove_digits)
    ngram_model = NgramModel.join_all(models)
    ngram_probabilities = ngram_model.ngram_probabilities
    chars = list(set(ngram_probabilities[-1]))

    random.shuffle(chars)
    code = {c: i for i, c in enumerate(chars)}

    mystery = load_text(args.mystery)[0]

    encoded_mystery = [code[c] for c in mystery]

    print("Code:")
    print([c for i, c in sorted((i, c) for c, i in code.items())])

    # Decide on a sample length based on the number of possible
    # characters...  should be something like 2x or 3x smallest
    # sample, then that value ** 2, ** 3, and so on. ?
    # mystery_len = len(mystery)
    # mystery_loglen = math.log(mystery_len, 10)
    # mystery_chunks = [

    decoder = GibbsSampler(ngram_probabilities,
                           args.num_cycles, args.burn_in,
                           args.display_interval,
                           args.jump_probability)
    decoder.gibbs_cycle(encoded_mystery)

def clean(args):
    models = iter_models(args.model_file,
                         args.ngram_size,
                         uniform_ws=args.uniform_whitespace,
                         lower=args.lower,
                         remove_punct=args.remove_punctuation,
                         remove_digits=args.remove_digits)
    ngram_model = NgramModel.join_all(models)

    input_texts = (
        text_token_probability_prep(text, args) for filename in args.input
        for text in load_text(filename)
    )

    if args.antimodel_file is not None:
        models = iter_models(args.antimodel_file,
                             args.ngram_size,
                             uniform_ws=args.uniform_whitespace,
                             lower=args.lower,
                             remove_punct=args.remove_punctuation,
                             remove_digits=args.remove_digits)

        negative_model = NgramModel.join_all(models)
        classify = NgramModelPairwiseClassifier(negative_model,
                                                ngram_model,
                                                prior=0.5)
        pfunc = classify.classify_logprob
    else:
        pfunc = ngram_model.joint_probability

    token_ps = (
        text_token_probability(text,
                               pfunc,
                               remove_punct=args.remove_punctuation)
        for text in input_texts
    )

    p_tokens = ([(p / (len(t) + 1), t) for (t, p) in text]
                for text in token_ps)
    p_tokens_flat = ((p / (len(t) + 1), t)
                     for text in token_ps
                     for (t, p) in text)

    if args.output_format == 'allterms':
        clean_display_all(args, p_tokens_flat, -args.threshold)
    elif args.output_format == 'negterms':
        clean_display_neg(args, p_tokens_flat, -args.threshold)
    elif args.output_format == 'totals':
        clean_display_totals(args, p_tokens_flat, -args.threshold)
    elif args.output_format == 'stoplist':
        clean_save_stoplist(args, p_tokens_flat, -args.threshold)
    elif args.output_format == 'golist':
        clean_save_golist(args, p_tokens_flat, -args.threshold)
    elif args.output_format == 'tagged':
        clean_display_tagged(args, p_tokens, -args.threshold)
    elif args.output_format == 'stripped':
        clean_display_stripped(args, p_tokens, -args.threshold)
    elif args.output_format == 'save':
        clean_save_stripped(args, p_tokens, -args.threshold)
    elif args.output_format == 'chars':
        clean_display_chars(args, p_tokens, pfunc, -args.threshold)

def text_token_probability_prep(text, args):
    if args.remove_digits:
        text = strip_digits(text)
    return text

def text_token_probability(text, pfunc, remove_punct=True):
    # I suspect the `.strip()` calls are redundant here.
    # Remove and test once the code is more stable.
    if remove_punct:
        def prep(s):
            return strip_punctuation(s.strip().lower())
    else:
        def prep(s):
            return s.strip().lower()

    text = uniform_whitespace(text)
    text = text.split()
    tokens_pvals = ((token, pfunc(prep(token)))
                    for token in text)

    return [(t.strip(), p) for t, p in tokens_pvals if t.strip()]

def clean_display_all(args, p_token, threshold):
    print('All tokens, with log probabilities.')
    for p, token in p_token:
        if p > threshold:
            print('{:20.20} {:8.5f}'.format(token, p))
        else:
            print('{:20.20} {:8.5f}      ***'.format(token, p))
    print()

def clean_display_neg(args, p_token, threshold):
    negterms = sorted((p, t) for p, t in p_token if p <= threshold)
    negterms.reverse()
    for p, token in negterms:
        print('{:20.20} {:8.5f}'.format(token, p))
    print()

def clean_display_totals(args, p_token, threshold):
    counts = [collections.defaultdict(int), collections.defaultdict(int)]
    token_probs = collections.defaultdict(list)
    for p, t in p_token:
        counts[p > threshold][t] += 1
        token_probs[t].append(p)

    neg_count, pos_count = counts
    tok_avg = {t: sum(p) / len(p) for t, p in token_probs.items()}

    pos_set = set(pos_count)
    neg_set = set(neg_count)
    both_set = pos_set & neg_set
    all_set = pos_set | neg_set

    print('{} total tokens'.format(sum(pos_count.values()) +
                                   sum(neg_count.values())))
    print('{} positive tokens'.format(sum(pos_count.values())))
    print('{} negative tokens'.format(sum(neg_count.values())))
    print()

    print('{} total types'.format(len(all_set)))
    print('{} positive types'.format(len(pos_set)))
    print('{} negative types'.format(len(neg_set)))
    print('{} overlapping types'.format(len(both_set)))
    print()

    neg_sorted = sorted(neg_set, key=tok_avg.get, reverse=True)
    neg_output = '\n'.join('{:40} {}'.format(t, tok_avg[t])
                           for t in neg_sorted[0:200])
    print('Most probable negative types:')
    print(neg_output)
    print()

def clean_save_stoplist(args, p_token, threshold, include_counts=True):
    stopwords = collections.defaultdict(int)
    for p, t in p_token:
        if p <= threshold:
            stopwords[t] += 1

    stopwords = [(c, t) for t, c in stopwords.items()]
    stopwords.sort(reverse=True)
    with open('stopwords.txt', 'w', encoding='utf-8') as out:
        for c, t in stopwords:
            if include_counts:
                t = t.replace('"', '""')
                out.write('"{}"'.format(t))
                out.write(',')
                out.write(str(c))
            else:
                out.write(t)
            out.write('\n')

def clean_save_golist(args, p_token, threshold, include_counts=True):
    gowords = collections.defaultdict(int)
    for p, t in p_token:
        if p > threshold:
            gowords[t] += 1

    gowords = [(c, t) for t, c in gowords.items()]
    gowords.sort(reverse=True)
    with open('gowords.txt', 'w', encoding='utf-8') as out:
        for c, t in gowords:
            if include_counts:
                t = t.replace('"', '""')
                out.write('"{}"'.format(t))
                out.write(',')
                out.write(str(c))
            else:
                out.write(t)
            out.write('\n')

def clean_display_tagged(args, p_orig, threshold):
    for i, p_o in enumerate(p_orig):
        tokens = ['{} '.format(o.strip()) if p > threshold else
                  '<<{}:{:6.3}>> '.format(o.strip(), p)
                  for p, o in p_o]
        text = ''.join(tokens)
        print('Document {}:'.format(i))
        print()
        for line in textwrap.wrap(text):
            print(line)
        print()

def clean_display_stripped(args, p_orig, threshold):
    for i, p_o in enumerate(p_orig):
        tokens = [o for p, o in p_o if p > threshold]
        text = ' '.join(tokens)
        for line in textwrap.wrap(text):
            print(line)
        print()
        print('Document {}:'.format(i))
        print()

def seq_pvals(token, pfunc):
    return [pfunc(token[:i]) / i
            for i in range(1, len(token) + 1)]

def tag_worst_char(token, pvals, threshold):
    # i = min(range(len(pvals)), key=pvals.__getitem__)
    i = max(range(len(pvals)), key=lambda i: pvals[i] <= threshold)
    return '{}{{{}}}{}'.format(token[0:i], token[i:i + 1], token[i + 1:])

def char_tag(token, pfunc, threshold):
    vals = seq_pvals(token, pfunc)
    return '{}:{}'.format(tag_worst_char(token, vals, threshold), vals)

def clean_display_chars(args, p_orig, pfunc, threshold):
    for i, p_o in enumerate(p_orig):
        tokens = ['{} '.format(o) if p > threshold else
                  '<<{} -- {}>> '.format(char_tag(o, pfunc, threshold), p)
                  for p, o in p_o]
        text = ''.join(tokens)
        print('Document {}:'.format(i))
        print()
        for line in textwrap.wrap(text):
            print(line)
        print()

def clean_save_stripped(args, p_orig, threshold):
    for i, (filename, p_o) in enumerate(zip(args.input, p_orig)):
        tokens = [o for p, o in p_o if p > threshold]
        text = ' '.join(tokens)
        filename, ext = os.path.splitext(filename)
        filename = '{}-stripped{}'.format(filename, ext)
        with open(filename, 'w', encoding='utf-8') as out:
            for line in textwrap.wrap(text):
                out.write(line)
                out.write('\n')

def save_model(args):
    all_models = iter_models(args.model_file,
                             args.ngram_size,
                             uniform_ws=args.uniform_whitespace,
                             lower=args.lower,
                             remove_punct=args.remove_punctuation,
                             remove_digits=args.remove_digits)

    model = NgramModel.join_all(all_models)

    if args.test_model:
        counts = model.to_counts()
        newmodel = NgramModel.from_counts(counts)
        if counts != model.to_counts():
            print('Destructive side-effect detected:')
            print('model.to_counts() != model.to_counts()')
        elif model != newmodel:
            print('Model was incorrectly reconstituted.')
            print('Results:')
            print()
            print('model.n: {}'.format(model.n))
            print('newmodel.n: {}'.format(newmodel.n))
            print()
            print('model.length: {}'.format(model.length))
            print('newmodel.length: {}'.format(newmodel.length))
            print()
            print('model.ngram_probabilities == '
                  'newmodel.ngram_probabilities: '
                  '{}'.format(model.ngram_probabilities ==
                              newmodel.ngram_probabilities))
        else:
            model.save(args.output)
    else:
        model.save(args.output)

def parse_args():
    parser = argparse.ArgumentParser(
        description='A set of tools based on a '
        'Naive Bayes charater model that does a decent job of distinguishing '
        'non-words from words given an input text and one or more model texts. '
        'These tools require no preprocessing or tokenization, and so can work '
        'with very messy data or unknown character sets.'
    )

    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        '-n', '--ngram-size', type=int, default=5,
        metavar='number', choices=[2, 3, 4, 5, 6],
        help='The number of characters to use for the ngram model. Defaults '
        'to 5. Models larger than 6 tend to underperform.'
    )
    model_parser.add_argument(
        '-m', '--model-file', type=str, required=True, metavar='filename',
        action='append', help='The name of a file to serve as a training '
        'model for character n-gram frequency analysis. May be used multiple '
        'times, in which case the models will be joined together, and must '
        'be used at least once. This may be a plain text file or a model '
        'that has been precompiled using the `save` command.'
    )
    model_parser.add_argument(
        '-a', '--antimodel-file', type=str, metavar='filename',
        action='append', help='The name of a file to serve as a negative '
        'model for character n-gram frequency analysis. Whereas model files '
        'are used to approximate a correct n-gram distribution, these files '
        'are used to approximate a specific incorrect n-gram distribution, '
        'which allows us to detect likely errors.'
    )
    model_parser.add_argument(
        '-l', '--lower', action='store_true', default=False,
        help='Convert all characters to lowercase before building models. '
        'Models that use only lower-case characters have better performance '
        'given the same number of characters, but they cannot handle '
        'capitalization gracefully, and so can only be used on input texts '
        'that have also been converted.'
    )
    model_parser.add_argument(
        '-p', '--remove-punctuation', action='store_true', default=False,
        help='Replace punctuation (as defined by `string.punctuation`) with '
        'whitespace prior to processing.'
    )
    model_parser.add_argument(
        '-d', '--remove-digits', action='store_true', default=False,
        help='Replace digits with whitespace prior to processing.'
    )
    model_parser.add_argument(
        '-w', '--uniform-whitespace', action='store_true', default=False,
        help='Make all whitespace uniform before building models. All '
        'whitespace characters or strings of whitespace-only characters will '
        'be replaced with a single space.'
    )

    commands = parser.add_subparsers(
        title='Available Commands',
        description='For more help, the -h/--help option for each command.',
    )

    gsd_parser = commands.add_parser(
        'decode', conflict_handler='resolve', parents=[model_parser],
        help='Use a modified version of Gibbs sampling to solve substitution '
        'ciphers. (The "Gibbs sampler" here may actually be closer to a '
        'metropolis-hastings sampler, because of the way it uses heuristics '
        'to escape local maxima.'
    )
    gsd_parser.add_argument(
        '-c', '--num-cycles', type=int, default=100,
        metavar='number', help='The total number of gibbs sampling cycles to '
        'complete.'
    )
    gsd_parser.add_argument(
        '-b', '--burn-in', type=int, default=20,
        metavar='number', help='The number of samples to discard before '
        'accumulating samples. Defaults to 20.'
    )
    gsd_parser.add_argument(
        '-i', '--display-interval', type=int, default=1,
        metavar='number', help='The frequency with which to display current '
        'key information.'
    )
    gsd_parser.add_argument(
        '-j', '--jump-probability', type=float,
        default=0.025, metavar='0.0-1.0', help='The probability of jumping '
        'from one state to another randomly. This is a heuristic for '
        'escaping local minima.'
    )
    gsd_parser.add_argument(
        'mystery', type=str, metavar='filename',
        help='The name of a file containing the text to be encrypted '
        'and decrypted.'
    )
    gsd_parser.set_defaults(command=decode)

    clean_parser = commands.add_parser(
        'clean', conflict_handler='resolve', parents=[model_parser],
        help='Use a Naive Bayes character sequence model to identify unlikely '
        'words.'
    )
    clean_parser.add_argument(
        '-o', '--output-format', type=str, default='tagged',
        choices=['allterms', 'negterms', 'totals', 'tagged',
                 'chars', 'stripped', 'save', 'stoplist', 'golist'],
        help='The output to generate. `allterms` displays all terms with '
        'negative terms marked. `negterms` displays only negative (non-word) '
        'terms, sorted from most to least likely. `tagged` displays the input '
        'text in its original order, but with negative terms tagged. '
        '`stripped` displays the input text with negative terms removed. '
        '`totals` displays total token and type counts, and a list of '
        'borderline terms. `save` saves the output of `stripped`. `stoplist` '
        'saves a list of negative terms for removal by another tool.'
    )
    clean_parser.add_argument(
        '-t', '--threshold', type=float, default=4,
        help='The likelihood threshold to be used for evaluation.'
    )
    clean_parser.add_argument(
        '-g', '--golist', type=str, metavar='filename', action='append',
        help='The name of a file containing words that should automatically '
        'be accepted as valid. May be used multiple times.'
    )
    clean_parser.add_argument(
        'input', type=str, metavar='filename',
        nargs='+',
        help='The name of a file or files containing the text to be processed.'
    )
    clean_parser.set_defaults(command=clean)

    save_parser = commands.add_parser(
        'save', conflict_handler='resolve', parents=[model_parser],
        help='Save an efficient copy of a Naive Bayes character '
        'sequence model.'
    )
    save_parser.add_argument(
        '-t', '--test-model', action='store_true', default=False,
        help='Test for correct conversion between probabilities and counts '
        'before saving. This verifies that saved models will be correctly '
        'restored in all contexts. (It also serves as a high-level unit '
        'test.)'
    )
    save_parser.add_argument(
        'output', type=str, metavar='filename',
        help='The name of the model save file.'
    )
    save_parser.set_defaults(command=save_model)

    args = parser.parse_args()
    if hasattr(args, 'command'):
        args.command(args)
    else:
        parser.print_help()

def main(argv):
    parse_args()

if __name__ == '__main__':
    main(sys.argv)
