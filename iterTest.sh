#!/bin/bash

make

train_file=../test.txt
embedding=../vectors_baseline.txt
save_vocab_to=/tmp/incre_vocab.txt
vocab_file=../vocab_baseline.txt
upList=../list.txt
output=/tmp/incre_embeddings


updateNew=1
updateAll=1
init=2

evaluate=1
binary=0

CBOW=1

LEARNING_RATE=0.05
WINDOW_SIZE=8
SAMPLE=1e-4
HS=0
NEGATIVE=25
Threads=4
minCount=5
debug=1
size=200


range=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 30 50 100 150 300 )

for iter in "${range[@]}"
do
./incre_train -train $train_file -output $output -save-vocab $save_vocab_to -read-vocab $vocab_file -embedding $embedding -min-count $minCount -threads $Threads -alpha $LEARNING_RATE -cbow $CBOW -window $WINDOW_SIZE -sample $SAMPLE -hs $HS -negative $NEGATIVE -updateNew $updateNew -init $init -update-list $upList -eval $evaluate -binary $binary -updateAll $updateAll -debug $debug -iter $iter
echo "finished iter $iter"

done



