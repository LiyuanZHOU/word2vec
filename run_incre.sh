#!/bin/bash

make

train_file=/media/data3tb2/NICTA_Clinical/wordEmbeddigTrain/i2b2+train.txt
embedding=/media/data3tb1/wordEmbedings/word2vec/NegSamp_n10/vectors_w5_v200
save_vocab_to=/media/data3tb2/NICTA_Clinical/wordEmbeddigTrain/wiki+i2b2+train_vocab.txt
vocab_file=/media/data3tb1/wordEmbedings/word2vec/vocab.txt
output=/media/data3tb2/NICTA_Clinical/wordEmbeddigTrain/wiki+i2b2+train_embed


updateNew=1
updateAll=0
init=2

evaluate=0
binary=0

CBOW=0

LEARNING_RATE=0.05
WINDOW_SIZE=5
SAMPLE=1e-3
HS=0
NEGATIVE=10
Threads=4
ITER=20
minCount=1

debug=2


#s=$(cut -d " " -f 1 $embedding | sed '2!d')
#if [ $s="</s>" ]; then                 
#	sed '/<\/s>/d' $embedding > $embedding;         
#fi
#s=$(cut -d " " -f 1 $vocab_file | sed '1!d')
#if [ $s="</s>" ]; then                 
#	sed '/<\/s>/d' $vocab_file > $vocab_file;         
#fi



#echo -train $train_file -output $output -save-vocab $save_vocab_to -read-vocab $vocab_file -embedding $embedding -min-count $minCount -threads $Threads -alpha $LEARNING_RATE -cbow $CBOW -iter $ITER -window $WINDOW_SIZE -sample $SAMPLE -hs $HS -negative $NEGATIVE -updateNew $updateNew -init $init -update-list $upList -eval $evaluate -binary $binary -updateAll $updateAll


./incre_train -train $train_file -output $output -save-vocab $save_vocab_to -read-vocab $vocab_file -embedding $embedding -min-count $minCount -threads $Threads -alpha $LEARNING_RATE -cbow $CBOW -iter $ITER -window $WINDOW_SIZE -sample $SAMPLE -hs $HS -negative $NEGATIVE -updateNew $updateNew -init $init -eval $evaluate -binary $binary -updateAll $updateAll -debug $debug
