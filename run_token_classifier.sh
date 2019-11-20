

##############################################################
#		EXECUTING MODELS FOR CONSTITUENCY
##############################################################



TEST_NAME="test"
DATA_DIR=./data/datasets/PTB-linearized/
MODEL_DIR=./data/bert_models_const/
PATH_GOLD_PARENTHESIZED=./data/datasets/PTB/$TEST_NAME.trees
LOGS=./data/logs_const/

if [ $TEST_NAME == "test" ]; then
    echo "DO_TEST"
    DO="--do_test"
else
    echo "DO_EVAL"
    DO="--do_eval"
fi

declare -a models=(
                   "bert.5e-4.lstms"
                   "bert.finetune.lstms"
		   "bert.5e-4.linear"
                   "bert.finetune.linear"
                  )

for i in "${models[@]}"
do

        echo "Model:" $i
        echo "Input:" $DATA_DIR
        echo "Gold test:" $PATH_GOLD_PARENTHESIZED
	MODEL_NAME=$i


	if [[ $MODEL_NAME == *"lstms" ]]; then
	   USE_BILSTMS="--use_bilstms"
	else
	   USE_BILSTMS=""
        fi

	OUTPUT_DIR=./data/outputs_const/$MODEL_NAME

	python run_token_classifier.py \
	--data_dir $DATA_DIR \
	--bert_model bert-base-cased \
	--task_name sl_tsv \
	--model_dir $MODEL_DIR/$MODEL_NAME \
	--output_dir $OUTPUT_DIR \
        --evalb_param True \
	--path_gold_parenthesized $PATH_GOLD_PARENTHESIZED \
	--parsing_paradigm constituency $DO $USE_BILSTMS --max_seq_length 250 > $LOGS/$MODEL_NAME.$TEST_NAME.log 2>&1

done





##############################################################
#        EXECUTING MODELS FOR DEPENDENCY PARSING
##############################################################



TEST_NAME="test"
PRED_SEG=true
MODEL_DIR=./data/bert_models_dep/
OUTPUT_DIR=./data/outputs_dep/
PATH_GOLD_CONLLU=./data/datasets/en-ewt/en_ewt-ud-$TEST_NAME.conllu
LOGS=./data/logs_dep/

if [ $TEST_NAME == "test" ]; then
    echo "DO_TEST"
    DO="--do_test"
else
    echo "DO_EVAL"
    DO="--do_eval"
fi


if [ $PRED_SEG = true ]; then
   echo "Using files with predicted segmentation"
   DATA_DIR=./data/datasets/EN_EWT-pred-linearized/
   PRED="_pred"
else
   DATA_DIR=./data/datasets/EN_EWT-linearized/
   echo "Using file with gold segmentation"
   PRED=""

fi

declare -a models=(
                   "bert.dep.5e-4.linear"
 		   "bert.dep.finetune.linear"
                   "bert.dep.5e-4.lstms"
                   "bert.dep.finetune.lstms"
                  )

for i in "${models[@]}"
do
        echo "Model:" $i
        echo "Input:" $DATA_DIR
        echo "Gold test:" $PATH_GOLD_CONLLU
	MODEL_NAME=$i

	if [[ $MODEL_NAME == *"lstms" ]]; then
	   USE_BILSTMS="--use_bilstms"
	else
	   USE_BILSTMS=""
        fi

	python run_token_classifier.py \
	--data_dir $DATA_DIR \
	--bert_model bert-base-cased \
	--task_name sl_tsv \
	--model_dir $MODEL_DIR/$MODEL_NAME \
	--max_seq_length 350 \
	--output_dir $OUTPUT_DIR/$MODEL_NAME$PRED.outputs.txt \
	--path_gold_conll $PATH_GOLD_CONLLU \
	--parsing_paradigm dependencies $DO $USE_BILSTMS --max_seq_length 350 > $LOGS/$MODEL_NAME.$TEST_NAME$PRED.log 2>&1



done













