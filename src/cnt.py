#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
from collections import defaultdict
import math
from const import TOKEN_TAG, SENTENCE_START, SENTENCE_END
from const import DEFAULT_NGRAM_CARDINALITY

"""Functions and class to count frequencies of n-grams in a file.

The data of the file must be formated in a particular way with one token per
line. A token can be a word or any other ponctuation symbol. An empty line
mark the end of a sentence.
This script simply count the number unigram, bigram and trigram. An n-gram is
a contiguous sequence of n tokens from a given sequence of text. An n-gram of
size 1 is referred to as a "unigram"; size 2 is a "bigram" (or, less
commonly, a "digram"); size 3 is a "trigram".
This is done thanks to a Hidden Markov Model.

@see: https://en.wikipedia.org/wiki/N-gram
"""


def usage():
    """Print a usage message."""
    print(
        "USAGE: python cnt.py [input_file]\n"
        "   Produce counts of tokens and n-grams from the input_file."
    )


def token_generator(tknsFile):
    """Create an iterator object for each token of the tokens file.

    The elements of the iterator are tuples containing a token and its
    associated tag. It looks like:
    C{(David, I-PER)}

    @note: Iterator elements representing empty lines are tuple with token
    and associated tag set to None.

    @warning: The input file must contain one token per line.

    @param tknsFile:
        The file containing the tokens.
    @type tknsFile: FILE

    @return: An iterator generating tuples of tokens and token tags.
    @rtype: generator
    """
    l = tknsFile.readline()
    while l:
        line = l.strip()
        if line:
            fields = line.split(" ")
            ne_tag = fields[-1]
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else:
            yield (None, None)
        l = tknsFile.readline()


def sentence_generator(toknIterator):
    """Create an iterator object for each sentence of the generated tokens.

    The elements of the iterator are lists of tuples. Every tuples are
    containing a token and its associated tag. It looks like:
    C{[('3.', 'O'), ('Goulnara', 'I-PER'), ('Fatkoullina', 'I-PER'),
    ('Russia', 'I-LOC')]}

    @param toknIterator:
        A generator iterating on each token of a file.
    @type toknIterator: generator

    @return: An iterator generating lists of tuples of tokens and token tags.
    @rtype: generator
    """
    current_sentence = []
    for l in toknIterator:
        if l == (None, None):
            if current_sentence:
                yield current_sentence
                current_sentence = []
            else:
                raise StopIteration
        else:
            current_sentence.append(l)
    if current_sentence:
        yield current_sentence


def ngram_generator(sntncIterator, n):
    """Create an iterator object for each n-gram generated from tokens lists.

    The elements of the iterator are tuples of tuples. Every sub-tuples are
    containing a token and its associated tag. It looks like:
    C{(('Division', 'O'), ('three', 'O'), (None, 'STOP'))}

    The sentence boundaries are added by inserting '*' and 'STOP' tokens. The
    elements of the generated iterator are all of length n.

    @param sntncIterator:
        A generator iterating on each sentence of a file.
    @type sntncIterator: generator

    @return: An iterator generating n-grams represented by tuple of n tuples
    including sentence boundaries tokens.
    @rtype: generator
    """
    for sent in sntncIterator:
        w_boundary = (n-1) * [(None, SENTENCE_START)]
        w_boundary.extend(sent)
        w_boundary.append((None, SENTENCE_END))
        ngrams = \
            (tuple(w_boundary[i:i+n]) for i in xrange(len(w_boundary)-n+1))
        for n_gram in ngrams:
            yield n_gram


class HMM(object):
    """Stores counts for n-grams and their probabilities."""

    def __init__(self, n=DEFAULT_NGRAM_CARDINALITY):
        """HMM creator.

        @param n:
            The n in n-gram: the n-gram cardinality.
        @type n: int
        """
        if n < 2:
            print('ERROR: N-gram cardinality must be 2 or more.',
                  file=sys.stderr)
        self.n = n
        self.emission_counts = defaultdict(int)
        self.wordCounts = defaultdict(int)
        self.ngramCounts = [defaultdict(int) for i in xrange(self.n)]
        self.states = set()
        self.words = set()

    def train(self, tknsFile):
        """Count n-grams frequencies and probabilities from a tokens file.

        This method compute the signle tokens counts and the n-grams counts
        using the tags.

        @param tknsFile:
            The file containing the tokens.
        @type tknsFile: FILE
        """
        ngram_iterator = ngram_generator(
            sentence_generator(token_generator(tknsFile)), self.n)
        for ngram in ngram_iterator:
            if len(ngram) != self.n:
                print('ERROR: Wrong n-gram cardinality (expected %i, get %i).'
                      % (self.n, len(ngram)), file=sys.stderr)
                sys.exit(10)
            tagsonly = tuple([ne_tag for word, ne_tag in ngram])
            for i in xrange(2, self.n+1):
                self.ngramCounts[i-1][tagsonly[-i:]] += 1

            if ngram[-1][0] is not None:
                self.ngramCounts[0][tagsonly[-1:]] += 1
                self.emission_counts[ngram[-1]] += 1
            if ngram[-2][0] is None:
                self.ngramCounts[self.n - 2][tuple((self.n - 1) *
                                                    [SENTENCE_START])] += 1

    def output_counts(self, output, printngrams=[1, 2, 3]):
        """Writes the n-grams counts on the output.

        @param output:
            The output where the results will be written.
        @type output: Stream
        @param printngrams:
            Format of the output: n-grams order.
        @type printngrams: list
        """
        for word, ne_tag in self.emission_counts:
            output.write(
                "%i %s %s %s\n" %
                (self.emission_counts[(word, ne_tag)], TOKEN_TAG, ne_tag, word))
        for n in printngrams:
            for ngram in self.ngramCounts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %
                             (self.ngramCounts[n-1][ngram], n, ngramstr))

    def load_counts(self, tknsFile):
        """Read n-grams counts from a tokens file.

        @param tknsFile:
            The file containing the tokens.
        @type tknsFile: FILE
        """
        self.n = 3
        self.emission_counts = defaultdict(int)
        self.wordCounts = defaultdict(int)
        self.ngramCounts = [defaultdict(int) for i in xrange(self.n)]
        self.states = set()
        self.words = set()

        for line in tknsFile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == TOKEN_TAG:
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.wordCounts[word] += count
                self.states.add(ne_tag)
                self.words.add(word)
            elif parts[1].endswith('GRAM'):
                n = int(parts[1].replace('-GRAM', ''))
                ngram = tuple(parts[2:])
                self.ngramCounts[n-1][ngram] = count

    def emission_prob(self, tkn, tag):
        """Compute emission probability.

        The emission probability is an algorithm of the Hidden Markov Model.

        @param tkn:
            A token.
        @type tkn: str
        @param tag:
            A named entity tag.
        @type tag: str

        @return: The computed emission probability for the given token and tag.
        @rtype: float
        """
        if tkn == SENTENCE_START:
            return float(1)
        tknTagCount = self.emission_counts[(tkn, tag)]
        tagCount = self.ngramCounts[0][(tag,)]
        return float(self.emission_counts[(tkn, tag)]) / tagCount

    def mle(self, trigram):
        """Compute the HMM maximum likelihood estimation.

        The maximum likelihood estimation (MLE) is an algorithm of the Hidden
        Markov Model.

        @param trigram:
            A list of 3 tokens.
        @type trigram: list

        @return: The computed maximum likelihood estimation for the given
            3-gram.
        @rtype: float

        @see: http://webcourse.cs.technion.ac.il/236522/Spring2008/ho/WCFiles/class08-m8.pdf
        """
        trigram = tuple(trigram)
        bigram = tuple(trigram[:2])
        tknTagCount = self.ngramCounts[2][trigram]
        if float(tknTagCount) == 0.0:
            return float(tknTagCount)
        else:
            tagCount = self.ngramCounts[1][bigram]
            return float(tknTagCount) / tagCount


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)
    try:
        input = file(sys.argv[1], 'r')
    except IOError:
        print("ERROR: Cannot read inputfile %s." % sys.argv[1], file=sys.stderr)
        sys.exit(1)
    counter = HMM(3)
    counter.train(input)
    counter.output_counts(sys.stdout)
