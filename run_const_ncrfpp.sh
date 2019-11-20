

TEST_NAME="test"
INPUT=./data/datasets/PTB-linearized/$TEST_NAME.tsv
TEST_PATH=./data/datasets/PTB/$TEST_NAME.trees
USE_GPU=True
EVALB=./tree2labels/EVALB/evalb
EVALB_PARAM=./tree2labels/EVALB/COLLINS.prm
OUTPUT=./data/outputs_const/
MODELS=./data/ncrfpp_models_const/
NCRFPP=./NCRFpp/
LOGS=./data/logs_const/


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
	python tree2labels/run_ncrfpp.py \
	--test $INPUT \
        --gold $TEST_PATH \
        --model $MODELS/$MODEL_NAME \
        --status test  \
        --gpu $USE_GPU \
        --output $OUTPUT/$MODEL_NAME.$TEST_NAME.outputs.txt \
        --evalb $EVALB \
        --evalb_param $EVALB_PARAM \
        --ncrfpp $NCRFPP > $LOGS/$MODEL_NAME.$TEST_NAME.$USE_GPU.log 2>&1

done


