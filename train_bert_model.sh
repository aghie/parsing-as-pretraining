

##############################################################
#		TRAIN MODEL FOR CONSTITUENTS
##############################################################


#BERT+linear finetuned
python run_token_classifier.py \
--data_dir ./data/datasets/PTB-linearized/ \
--bert_model bert-base-cased \
--task_name sl_tsv \
--model_dir /tmp/bert.PTB.finetune.linear.model \
--output_dir /tmp/dev.output \
--path_gold_parenthesized ./data/datasets/PTB/dev.trees \
--parsing_paradigm constituency --do_train --do_eval --num_train_epochs 15 --max_seq_length 250


#BERT+linear not finetuned
python run_token_classifier.py \
--data_dir ./data/datasets/PTB-linearized/ \
--bert_model bert-base-cased \
--task_name sl_tsv \
--model_dir /tmp/bert.PTB.linear.model \
--output_dir /tmp/dev.output \
--path_gold_parenthesized ./data/datasets/PTB/dev.trees \
--parsing_paradigm constituency --do_train --do_eval --learning_rate 5e-4 --num_train_epochs 15 --max_seq_length 250 --not_finetune



#BERT+BILSTM+linear finetuned
python run_token_classifier.py \
--data_dir ./data/datasets/PTB-linearized/ \
--bert_model bert-base-cased \
--task_name sl_tsv \
--model_dir /tmp/bert.PTB.finetune.lstms.model \
--output_dir /tmp/dev.output \
--evalb tree2labels/EVALB/evalb \
--path_gold_parenthesized ./data/datasets/PTB/dev.trees \
--tree2labels tree2labels/ \
--parsing_paradigm constituency --do_train --do_eval --num_train_epochs 15 --max_seq_length 250 --use_bilstms 




#BERT+BILSTM+linear not finetuned
python run_token_classifier.py \
--data_dir ./data/datasets/PTB-linearized-proof/ \
--bert_model bert-base-cased \
--task_name sl_tsv \
--model_dir /tmp/bert.PTB.lstms.model \
--output_dir /tmp/dev.output \
--evalb tree2labels/EVALB/evalb \
--path_gold_parenthesized ./data/datasets/PTB/dev.trees \
--tree2labels tree2labels/ \
--parsing_paradigm constituency --do_train --do_eval --learning_rate 5e-4 --num_train_epochs 15 --max_seq_length 250 --use_bilstms --not_finetune




##############################################################
#		TRAIN MODEL FOR DEPENDENCIES
##############################################################


#BERT+linear finetuned
python run_token_classifier.py \
--data_dir ./data/datasets/EN_EWT-linearized/ \
--bert_model bert-base-cased \
--task_name sl_tsv \
--model_dir /tmp/bert.EN_EWT.finetune.linear.model \
--output_dir /tmp/dev.output \
--path_gold_conll ./data/datasets/en-ewt/en_ewt-ud-dev.conllu \
--parsing_paradigm dependencies --do_train --do_eval --learning_rate 5e-4 --num_train_epochs 15 --max_seq_length 250


#BERT+linear not finetuned
python run_token_classifier.py \
--data_dir ./data/datasets/EN_EWT-linearized/ \
--bert_model bert-base-cased \
--task_name sl_tsv \
--model_dir /tmp/bert.EN_EWT.linear.model \
--output_dir /tmp/dev.output \
--path_gold_conll ./data/datasets/en-ewt/en_ewt-ud-dev.conllu \
--parsing_paradigm dependencies --do_train --do_eval --learning_rate 5e-4 --num_train_epochs 15 --not_finetune --max_seq_length 250


#BERT+BILSTM+linear finetuned
python run_token_classifier.py \
--data_dir ./data/datasets/EN_EWT-linearized/ \
--bert_model bert-base-cased \
--task_name sl_tsv \
--model_dir /tmp/bert.EN_EWT.finetune.lstms.model \
--output_dir /tmp/dev.output \
--path_gold_conll ./data/datasets/en-ewt/en_ewt-ud-dev.conllu \
--parsing_paradigm dependencies --do_train --do_eval --num_train_epochs 15 --use_bilstms --max_seq_length 250




#BERT+BILSTM+linear not finetuned
python run_token_classifier.py \
--data_dir ./data/datasets/EN_EWT-linearized/ \
--bert_model bert-base-cased \
--task_name sl_tsv \
--model_dir /tmp/bert.EN_EWT.lstms.model \
--output_dir /tmp/dev.output \
--path_gold_conll ./data/datasets/en-ewt/en_ewt-ud-dev.conllu \
--parsing_paradigm dependencies --do_train --do_eval --learning_rate 5e-4 --num_train_epochs 15 --use_bilstms --not_finetune --max_seq_length 250









