#!/bin/bash

INT_RE='^[0-9]+$'
SMALL_LETTER_RE='^[a-z]+$'
FLOAT_RE='^[0-9]+([.][0-9]+)?$'
DATASET="eng"
TEST="a"
RES_DIR="results"
SRC_DIR="src"
DATA_DIR="data"

if [ ! -d "$RES_DIR" ]; then
    mkdir $RES_DIR
fi

usage(){
    echo -e "USAGE:\trunme [-d -t]"
    echo -e "OPTIONS:"
    echo -e "\t-d or --dataset specify the dataset to use, it must be the"
    echo -e "\t\tname of a folder inside the 'data' folder"
    echo -e "\t-t or --test specify the test files to use, it is a small letter"
    echo -e "\t\t'a' (testa) is mandatory and is the default value"
}

while test $# -gt 0; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -d*|--dataset*)
            shift
            if [ ! -d $DATA_DIR/$1 ]; then
              echo "ERROR: DATASET must correspond to an existing folder inside 'data'" >&2
              exit 1
            fi
            DATASET=$1
            shift
            ;;
        -t*|--test*)
            shift
            if ! [[ $1 =~ $SMALL_LETTER_RE ]]; then
               echo "ERROR: TEST must be a letter between 'a' and 'z'" >&2
               exit 1
            fi
            TEST=$1
            shift
            ;;
        *)
            break
            ;;
    esac
done

cp $DATA_DIR/$DATASET/$DATASET.train $RES_DIR/train.truncated
python $SRC_DIR/cnt.py $DATA_DIR/$DATASET/$DATASET.train > $RES_DIR/ngrams.counts
python $SRC_DIR/fltr.py $RES_DIR/ngrams.counts $RES_DIR/train.truncated
python $SRC_DIR/cnt.py $RES_DIR/train.truncated > $RES_DIR/ngrams.truncated.counts
python $SRC_DIR/tag.py $RES_DIR/ngrams.truncated.counts $DATA_DIR/$DATASET/$DATASET.test$TEST > $RES_DIR/predicted.tags
python $SRC_DIR/cmp.py $DATA_DIR/$DATASET/$DATASET.test$TEST.orig $RES_DIR/predicted.tags

exit 0
