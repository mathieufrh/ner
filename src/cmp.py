#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
from const import TAG_CLASSES, TAG_IN_PREFIX, TAG_BOUNDARY_PREFIX, TAG_NONE

"""Compare the predicted tags to the original tags.

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

These functions compare the named entities predicted by the program to the
original named entities of the dataset. The precision column is the
percentage of correct tags in every tags found by the program. The recall
column is the percentage of correct tags in every tags of the original
dataset found by the program. The F column is a ratio computed by
F = 2 * precision * recall / (precision + rrecall).

@see: http://www.cnts.ua.ac.be/conll2002/ner/
"""


def usage():
    """Print a usage message."""
    print(
        "USAGE: python eval_ne_tagger.py [key_file] [prediction_file]"
        "    Evaluate the NE-tagger output in prediction_file against"
        "    the gold standard in key_file. Output accuracy, precision,"
        "    recall and F1-Score for each NE tag type.\n"
    )


def token_generator(tknsFile, addProb=False):
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
    tagfield = addProb and -2 or -1
    try:
        while l:
            line = l.strip()
            if line:
                fields = line.split(' ')
                tag = fields[tagfield]
                word = ' '.join(fields[:tagfield])
                yield word, tag
            else:
                yield (None, None)
            l = tknsFile.readline()
    except IndexError:
        print("Could not read line: %s" % line, file=sys.stderr)
        sys.exit(1)


class EntityCounter(object):
    """Counts for each named enity class."""

    def __init__(self):
        """EntityCounter Creator"""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def get_precision(self):
        """Compute precision of the predicted named entities."""
        return self.tp / float(self.tp + self.fp)

    def get_recall(self):
        """Compute recall rate of the predicted named entities."""
        return self.tp / float(self.tp + self.fn)

    def get_accuracy(self):
        """Compute accuracy of the predicted named entities."""
        return (
            self.tp + self.tn) / float(self.tp + self.tn + self.fp + self.fn)


class Comparator(object):
    """Compare the predicted named entities with the original named entities."""

    def __init__(self):
        """Comparator Creator"""
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.class_counts = {}
        for c in TAG_CLASSES:
            self.class_counts[c] = EntityCounter()

    def check_end(self, currPredType, currOrigType, predTag, origTag, predType,
                  origType):
        """Check if we reach the end of a named entity.

        A named entity can end at the end boundarie of the sentence, if the
        current token tag is O or if a new entoty starts. We know a new
        entity starts if its tag is B-* or I-* with * different than the
        actual tag class.

        @param currPredType:
            Current prediction dataset named entity class.
        @type currPredType: str
        @param currOrigType:
            Current original dataset named entity class.
        @type currOrigType: str
        @param predTag:
            Current prediction dataset named entity tag.
        @type predTag: str
        @param origTag:
            Current original dataset named entity tag.
        @type origTag: str
        @param predType:
            Prediction token named entity class.
        @type predType: str
        @param origType:
            Original token named entity class.
        @type origType: str

        @return: True weither the prediction/original named entity ends.
        @rtype: bool
        """
        predEnds = \
            currPredType is not None and \
            ((predTag is None or
                predTag[0] in "OB") or
                (currPredType != predType and
                    predTag[0] == TAG_IN_PREFIX))
        origEnds = \
            currOrigType is not None and \
            ((origTag is None or
                origTag[0] in "OB") or
                (currOrigType != origType and
                    origTag[0] == TAG_IN_PREFIX))
        return predEnds, origEnds

    def check_start(self, currPredType, currOrigType, predTag, origTag,
                    predType, origType, predTkn):
        """Check if we reach the end of a named entity.

        A named entity can start during a sentence or if tag is I and last
        one was O or if it is I and the tag class is diffrent.

        @param currPredType:
            Current prediction dataset named entity class.
        @type currPredType: str
        @param currOrigType:
            Current original dataset named entity class.
        @type currOrigType: str
        @param predTag:
            Current prediction dataset named entity tag.
        @type predTag: str
        @param origTag:
            Current original dataset named entity tag.
        @type origTag: str
        @param predType:
            Prediction token named entity class.
        @type predType: str
        @param origType:
            Original token named entity class.
        @type origType: str
        @param predTkn:
            Prediction token.
        @type predTkn: str

        @return: True weither the prediction/original named entity starts.
        @rtype: bool
        """
        if predTkn is not None:
            predStarts = \
                (predTag is not None and
                    predTag[0] == TAG_BOUNDARY_PREFIX) or \
                (currPredType is None and
                    predTag[0] == TAG_IN_PREFIX) or \
                (currPredType is not None and
                    currPredType != predType and
                    predTag.startswith(TAG_IN_PREFIX))
            origStarts = \
                (origTag is not None and
                    origTag[0] == TAG_BOUNDARY_PREFIX) or \
                (currOrigType is None and
                    origTag[0] == TAG_IN_PREFIX) or \
                (currOrigType is not None and
                    currOrigType != origType and
                    origTag.startswith(TAG_IN_PREFIX))
        else:
            return False, False
        return predStarts, origStarts

    def compare(self, originalTknsTags, predictionTknsTags):
        """Compare the predicted and original named entities.

        @param originalTknsTags:
            Generator of token/tag tuples.
        @type originalTknsTags: generator
        @param predictionTknsTags:
            Generator of token/tag tuples.
        @type predictionTknsTags: generator
        """
        currPredType = None
        currPredStart = None
        currOrigType = None
        currOrigStart = None
        total = 0
        for origTkn, origTag in originalTknsTags:
            predTkn, predTag = predictionTknsTags.next()
            if origTkn != predTkn:
                print('Original and prediction fils do not correspond.')
                print(origTkn)
                print(predTkn)
                print('Exiting now...')
                sys.exit(1)
            origType = origTag is None and TAG_NONE or origTag.split("-")[-1]
            predType = predTag is None and TAG_NONE or predTag.split("-")[-1]
            predEnds, origEnds = self.check_end(
                currPredType, currOrigType, predTag, origTag, predType,
                origType)
            predStarts, origStarts = self.check_start(
                currPredType, currOrigType, predTag, origTag, predType,
                origType, predTkn)
            if origEnds and predEnds:
                if currOrigStart == currPredStart and \
                        currOrigType == currPredType:
                    self.tp += 1
                    self.class_counts[currPredType].tp += 1
                else:
                    self.fp += 1
                    self.fn += 1
                    self.class_counts[currPredType].fp += 1
                    self.class_counts[currOrigType].fn += 1
            elif origEnds:
                self.fn += 1
                self.class_counts[currOrigType].fn += 1
            elif predEnds:
                self.fp += 1
                self.class_counts[currPredType].fp += 1
            elif currPredType is None and currPredType is None:
                self.tn += 1
                for c in TAG_CLASSES:
                    self.class_counts[c].tn += 1
            if origEnds:
                currOrigType = None
            if predEnds:
                currPredType = None
            if origStarts:
                currOrigStart = total
                currOrigType = origType
            if predStarts:
                currPredStart = total
                currPredType = predType
            total += 1

    def print_res_table(self):
        """Compute the performance of the program.

        This function compute the precision, recall and F value of the predicted
        named entities for each tag class and the average value. The results are
        written in a table on the output.
        """
        if self.tp + self.tn + self.fp + self.fn == 0:
            acc = 1
        else:
            acc = (self.tp + self.tn) / float(
                self.tp + self.tn + self.fp + self.fn)
        if self.tp+self.fp == 0:
            avgPrec = 1
        else:
            avgPrec = self.tp / float(self.tp + self.fp)
        if self.tp+self.fn == 0:
            avgRec = 1
        else:
            avgRec = self.tp / float(self.tp + self.fn)
        print('class\t| precision\t| recall\t| F')
        print('--------+---------------+---------------+------')
        avgFscore = (2 * avgPrec * avgRec) / (avgPrec + avgRec)
        for c in TAG_CLASSES:
            c_tp = self.class_counts[c].tp
            c_tn = self.class_counts[c].tn
            c_fp = self.class_counts[c].fp
            c_fn = self.class_counts[c].fn
            if (c_tp + c_tn + c_fp + c_fn) == 0:
                c_acc = 1
            else:
                c_acc = (c_tp + c_tn) / float(c_tp + c_tn + c_fp + c_fn)
            if (c_tp + c_fn) == 0:
                print('WARNING: entity type of %s is missing in original'
                      'dataset.' % c, file=sys.stderr)
                rec = 1
            else:
                rec = c_tp / float(c_tp + c_fn)
            if (c_tp + c_fp) == 0:
                print('WARNING: entity type of %s is missing in prediction'
                      'dataset.' % c, file=sys.stderr)
                prec = 1
            else:
                prec = c_tp / float(c_tp + c_fp)

            if prec + rec == 0:
                fscore = 0
            else:
                fscore = (2*prec * rec) / (prec + rec)
            print("%s\t| %.2f%%\t| %.2f%%\t| %.2f" % (
                c, prec * 100, rec * 100, fscore * 100))
        print('--------+---------------+---------------+------')
        print('AVERAGE\t| %.2f%%\t| %.2f%%\t| %.2f' % (
            avgPrec * 100, avgRec * 100, avgFscore * 100))
        print('\n' + '-' * 47)
        print('Original dataset contains %i named entitied.' % (
            self.tp + self.fn))
        print('Computed results contain %i named entitied.' % (
            self.tp + self.fp))
        print('-' * 47)
        print('%i computed named entitied match the original named entities.\n'
              % self.tp)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        sys.exit(1)
    origIterator = token_generator(file(sys.argv[1]))
    predIterator = token_generator(file(sys.argv[2]), addProb=True)
    evaluator = Comparator()
    evaluator.compare(origIterator, predIterator)
    evaluator.print_res_table()
