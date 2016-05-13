#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Functions to replace specific categories of words in a tokens/tags file.

The data of the file must be formated in a particular way: the CoNLL 2002
format. The data consists of two columns separated by a single space. Each
word has been put on a separate line and there is an empty line after each
sentence. The first item on each line is a word and the second the named
entity tag. The tags have the same format as in the chunking task: a B
denotes the first item of a phrase and an I any non-initial word. There are
four types of phrases: person names (PER), organizations (ORG), locations
(LOC) and miscellaneous names (MISC). Here is an example:
C{        Wolff B-PER}
C{            , O}
C{    currently O}
C{            a O}
C{   journalist O}
C{           in O}
C{    Argentina B-LOC}
C{            , O}
C{       played O}
C{         with O}
C{          Del B-PER}
C{       Bosque I-PER}
C{           in O}
C{          the O}
C{        final O}
C{        years O}
C{           of O}
C{          the O}
C{    seventies O}
C{           in O}
C{         Real B-ORG}
C{       Madrid I-ORG}
C{            . O}

These functions are used to replace every instances of words belonging to
specific categories by remplacement GROUP TOKENS. The following table show
the word categories along with their remplacement GROUP TOKENS::
           CATEGORY           |  GROUP TOKEN
    --------------------------+-------------
         Uncommon words       |  _UNCOMMON_
    Words starting with a cap |_PROPER_NOUN_
       Capitalized words      |_CAPITALIZED_
     Punctuation or numbers   |_PUNCTUATION_

@see: http://www.cnts.ua.ac.be/conll2002/ner/
"""

from __future__ import print_function
import sys
import math
import string
import fileinput
from collections import defaultdict
from cnt import HMM
from const import UNCOMMON, PROPER_NOUN, CAPITALIZED, PUNCTUATION


def usage():
    """Print a usage message."""
    print(
        "USAGE: python group_rare.py [counts_file] [input_file]"
        "    Read in named entity tagged training input file"
        "    and corresponding counts_file and group words"
        "    based on defined criteria. Replace all grouped"
        "    words in the training symbol for a common symbol"
        "    for said group in the form _GROUPID_."
    )


def tkn_cap_first(tkn):
    """Check if a token starts with a capitalized letter.

    @param tkn:
        The token to check.
    @type tkn: str

    @return: True or False wether the token do starts with a capitalized letter.
    @rtype: bool
    """
    return tkn.istitle()


def tkn_all_caps(tkn):
    """Check if a token contains only capitalized letters.

    @param tkn:
        The token to check.
    @type tkn: str

    @return: True or False wether the token contains only capitalized letters.
    @rtype: bool
    """
    return tkn.isupper()


def tkn_num_punct(tkn):
    """Check if a token contains only punctuation or digits.

    @param tkn:
        The token to check.
    @type tkn: str

    @return: True or False wether the token contains only punctuation/digits.
    @rtype: bool
    """
    return all(c in string.punctuation or c.isdigit() for c in tkn)


def remove_sub_dict(subdict, dictionary):
    """Remove a list of keys from a given dictionary.

    @param subdict:
        The sub-dictionary containing the entries to remove from the other
        dictionary.
    @type subdict: dict
    @param dictionary:
        The original dictionary from which the entries will be removed.
    @type dictionary: dict
    """
    for key in subdict:
        dictionary.pop(key, None)


def replace_all(toknTagFile, wordsToReplace, substitute):
    """Replace all instances of every words of a list with subsitute in a file.

    @param toknTagFile:
        The path of the file containing token/tag associations which will be
        modified.
    @type toknTagFile: str
    @param wordsToReplace:
        The dictionary containing the words to replace as keys.
    @type wordsToReplace: dict
    @param substitute:
        The substitution string which will replace every instances of of
        every words in 'wordsToReplace'.
    @type substitute: str
    """
    for line in fileinput.input(toknTagFile, inplace=True):
        parts = line.strip().split(" ")
        if parts[0] in wordsToReplace:
            parts[0] = substitute
            line = " ".join(parts) + "\n"
        sys.stdout.write(line)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        sys.exit(1)
    try:
        input = file(sys.argv[1], 'r')
    except IOError:
        print(
            'ERROR: Cannot read input file %s.' % sys.argv[1], file=sys.stderr)
        sys.exit(1)
    try:
        output = sys.argv[2]
    except IOError:
        print(
            'ERROR: Cannot read input file %s.' % sys.argv[2], file=sys.stderr)
        sys.exit(1)
    counter = HMM(3)
    counter.load_counts(input)
    uncommon = dict((k, v) for k, v in counter.wordCounts.iteritems() if v < 5)
    common = dict((k, v) for k, v in counter.wordCounts.iteritems() if v > 5)
    cf = dict((k, v) for k, v in uncommon.iteritems() if tkn_cap_first(k))
    np = dict((k, v) for k, v in uncommon.iteritems() if tkn_num_punct(k))
    ac = dict((k, v) for k, v in uncommon.iteritems() if tkn_all_caps(k))
    remove_sub_dict(cf, uncommon)
    remove_sub_dict(np, uncommon)
    remove_sub_dict(ac, uncommon)
    replace_all(output, uncommon, UNCOMMON)
    replace_all(output, cf, PROPER_NOUN)
    replace_all(output, np, PUNCTUATION)
    replace_all(output, ac, CAPITALIZED)
