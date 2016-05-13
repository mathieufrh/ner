#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import operator
import math
import itertools
import string
from collections import defaultdict
from cnt import HMM
from const import UNCOMMON, PROPER_NOUN, CAPITALIZED, PUNCTUATION
from const import TOKEN_TAG, SENTENCE_START, SENTENCE_END, UNCOMMON_LIMIT


def usage():
    """Print a usage message."""
    print(
        "python viterbi_grouped.py [counts_file] [testFile] > [output_file]"
        "    Read in counts_file generated from training set and"
        "    test set of data, then predict tags for each word in"
        "    test set."
        "    Modification on original viterbi.py to use grouped words from _Q6_"
        "    Results are stored in the following format:"
        "    <word> <tag> <log probability of tagged sequence up to this word>"
    )


def token_generator(tknsFile):
    """Create an iterator object for each token of the tokens file.

    The elements of the iterator are tokens of the file as strings. It looks
    like:
    C{sidelines}

    @note: Iterator elements representing empty lines are None.

    @warning: The input file must contain one token per line.

    @param tknsFile:
        The file containing the tokens.
    @type tknsFile: FILE

    @return: An iterator generating tokens as strings.
    @rtype: generator
    """
    l = tknsFile.readline()
    while l:
        line = l.strip()
        if line:
            yield line
        else:
            yield None
        l = tknsFile.readline()


def sentence_generator(toknIterator):
    """Create an iterator object for each sentence of the generated tokens.

    The elements of the iterator are lists of strings. It looks like:
    C{['They', 'said', 'there', 'was', 'still', 'demand', 'for', 'blue',
    'chips', '.']}

    @param toknIterator:
        A generator iterating on each token of a file.
    @type toknIterator: generator

    @return: An iterator generating lists of tokens.
    @rtype: generator
    """
    currSntnc = []
    for l in toknIterator:
        if l is None:
            if currSntnc:
                yield currSntnc
                currSntnc = []
            else:
                print('WARNING: Got empty input file/stream.', file=sys.stderr)
                raise StopIteration
        else:
            currSntnc.append(l)
    if currSntnc:
        yield currSntnc


def viterbi(sentence):
    """Viterbi alorithm for finding the mst likely tag for every tokens.

    The Viterbi algorithm is a dynamic programming algorithm for finding the
    most likely sequence of hidden states – called the Viterbi path – that
    results in a sequence of observed events, especially in the context of
    Markov information sources and hidden Markov models.

    The function has been implemented based on the following algorithm:
    Input:
    ======
    Z=z1,z2,...,zn    the input observed sequence

    Initialization:
    ===============
    k=1               time index
    S(c1)=c1
    L(c1)=0           variable accumulating the lengths, the initial length is 0

    Recursion:
    ==========
    For all transitions tk=(ck,ck+1)
    ....compute: L(ck,ck+1) = L(ck) + l [tk = (ck,ck+1)] among all ck.
    Find L(ck+1) = min L(ck,ck+1)
    For each ck+1
    ....store L(ck+1) and the corresponding survivor S(ck+1).
    k=k+1
    Repeat until k=n

    @param sentence:
        A list of tokens (strings) representing a sentence.
    @type sentence: list

    @see: https://en.wikipedia.org/wiki/Viterbi_algorithm
    @see: http://courses.washington.edu/ling570/gina_fall11/slides/ling570_class12_viterbi.pdf
    """
    n = len(sentence)
    padSntnc = (2) * [SENTENCE_START]
    padSntnc.extend(sentence)
    padSntnc.append(SENTENCE_END)
    K = [SENTENCE_START] + (n) * [counter.states] + [SENTENCE_START]
    pi = [defaultdict(float) for i in xrange(n + 1)]
    pi[0][(SENTENCE_START, SENTENCE_START)] = 1.0
    for k in xrange(1, n + 1):
        word = padSntnc[k + 1]
        original_word = padSntnc[k + 1]
        if word not in counter.words or \
                counter.wordCounts[word] < UNCOMMON_LIMIT:
            if word.isupper():
                word = CAPITALIZED
            elif word.istitle():
                word = PROPER_NOUN
            elif all(c in string.punctuation or c.isdigit() for c in word):
                word = PUNCTUATION
            else:
                word = UNCOMMON
        for u in K[k - 1]:
            for v in K[k]:
                candidates = defaultdict(float)
                for w in K[k - 2]:
                    candidates[w] = \
                        pi[k - 1][(w, u)] * \
                        counter.mle([w, u, v]) * \
                        counter.emission_prob(word, v)
                final = max(
                    candidates.iteritems(), key=operator.itemgetter(1))
                pi[k][(u, v)] = final[1]
        final_k_idx = max(pi[k].iteritems(), key=operator.itemgetter(1))
        prob = final_k_idx[1]
        if prob == 0:
            logProb = 0
        else:
            logProb = math.log(prob)
        sys.stdout.write(
            '%s %s %s\n' % (original_word, final_k_idx[0][1], logProb))
    finalCandidates = defaultdict(float)
    perms = itertools.product(K[n - 1], K[n])
    for perm in perms:
        finalCandidates[perm] = \
            pi[n][perm] * \
            counter.mle(list(perm + (SENTENCE_END,)))
    final_sent_prob_idx = max(
        finalCandidates.iteritems(), key=operator.itemgetter(1))
    print('')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        sys.exit(1)
    try:
        counts_file = file(sys.argv[1], 'r')
    except IOError:
        print(
            'ERROR: Cannot read input file %s.' % sys.argv[1], file=sys.stderr)
        sys.exit(1)
    try:
        testFile = file(sys.argv[2], 'r')
    except IOError:
        print(
            'ERROR: Cannot read input file %s.' % sys.argv[2], file=sys.stderr)
        sys.exit(1)
    counter = HMM(3)
    counter.load_counts(counts_file)
    sntncIterator = sentence_generator(token_generator(testFile))
    for sentence in sntncIterator:
        viterbi(sentence)
