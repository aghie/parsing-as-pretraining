TEST_NAME="test"
PRED_SEG=false
TEST_PATH=./data/datasets/en-ewt/en_ewt-ud-$TEST_NAME.conllu
USE_GPU=True
OUTPUT=./data/outputs_dep/
MODELS=./data/ncrfpp_models_dep
NCRFPP=./NCRFpp/
CONLL_UD=./dep2labels/conll17_ud_eval.py
LOGS=./data/logs_dep/


if [ $PRED_SEG = true ]; then
   echo "Using files with predicted segmentation"
   PRED="_pred"
   INPUT=./data/datasets/EN_EWT-pred-linearized/$TEST_NAME.tsv
else
   echo "Using file with gold segmentation"
   PRED=""
   INPUT=./data/datasets/EN_EWT-linearized/$TEST_NAME.tsv

fi



declare -a models=(
                          "glove.840B.300.finetune.linear"
                          "glove.840B.300.linear"
                          "glove.840B.300.finetune.lstms"
                          "glove.840B.300.lstms"
                          "random.300.linear"
                          "random.300.lstms"
                          "random.300.finetune.linear"
                          "random.300.finetune.lstms"
                          "random.1024.linear"
                          "random.1024.lstms"
                          "random.1024.finetune.linear"
                          "random.1024.finetune.lstms"
                          "wiki-news-300d-1M.finetune.linear"
                          "wiki-news-300d-1M.finetune.lstms"
                          "wiki-news-300d-1M.linear"
                          "wiki-news-300d-1M.lstms"
                          "GoogleNews-vectors-negative300.finetune.linear"
                          "GoogleNews-vectors-negative300.finetune.lstms"
                          "GoogleNews-vectors-negative300.linear"
                          "GoogleNews-vectors-negative300.lstms"
                          "dyer.sskip.100.linear"
                          "dyer.sskip.100.finetune.linear"
                          "dyer.sskip.100.finetune.lstms"
                          "dyer.sskip.100.lstms"
                          "elmo.finetune.linear"
                          "elmo.finetune.lstms"
                          "elmo.lstms"
                          "elmo.linear"
                        )


for i in "${models[@]}"
do
        echo "Model:" $i
        echo "Input:" $INPUT
        echo "Gold test:" $TEST_PATH
	MODEL_NAME=$i
	taskset --cpu-list 1 \
	python dep2labels/run_dep_ncrfpp.py \
	--test $INPUT \
        --gold $TEST_PATH \
        --model $MODELS/$MODEL_NAME \
        --status test  \
        --gpu $USE_GPU \
        --conll_ud $CONLL_UD \
        --output $OUTPUT/$MODEL_NAME.$TEST_NAME$PRED.outputs.conllu \
        --ncrfpp $NCRFPP > $LOGS/$MODEL_NAME.$TEST_NAME$PRED.$USE_GPU.log 2>&1

done


