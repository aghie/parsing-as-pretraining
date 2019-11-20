# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "tree2labels"))
import csv

import logging
import argparse
import random
import tempfile
import subprocess
from tqdm import tqdm, trange

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert import BertForTokenClassification, BertModel
from sklearn.metrics import accuracy_score
from tree2labels.utils import sequence_to_parenthesis

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class MyBertForTokenClassification(BertForTokenClassification):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels, finetune, use_bilstms=False):
        super(MyBertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.use_bilstms = use_bilstms
        
        if self.use_bilstms:
            self.lstm = nn.LSTM(config.hidden_size, 400, num_layers=2, batch_first=True, 
                                bidirectional=True)
            self.classifier =  nn.Linear(800, num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, num_labels)
            
        self.apply(self.init_bert_weights)
        self.finetune = finetune


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if not self.finetune:
            sequence_output = sequence_output.detach()        

        if self.use_bilstms:
            sequence_output, hidden = self.lstm(sequence_output, None)
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits



class InputSLExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, 
                 text_a_list,
                 text_a_postags, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the sentence
            label: (Optional) list. The labels for each token. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.text_a_list = text_a_list 
        self.text_a_postags = text_a_postags
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, labels_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels_ids = labels_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SLProcessor(DataProcessor):
    """Processor for PTB formatted as sequence labeling seq_lu file"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self, data_dir):
        """See base class."""
        
        train_samples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        dev_samples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        

        train_labels = [label for sample in train_samples 
                            for label in sample.labels]
        
        dev_labels = [label for sample in dev_samples 
                          for label in sample.labels]

        labels = []
        labels.append("[MASK_LABEL]")
        labels.append("-EOS-")
        labels.append("-BOS-")
        train_labels.extend(dev_labels)
        for label in train_labels:
            if label not in labels:
                labels.append(label)
        return labels
    
    
    def _preprocess(self, word):
        if word == "-LRB-": 
            word = "("
        elif word == "-RRB-": 
            word = ")"
        return word

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentences_texts = []
        sentences_postags = []
        sentences_labels = []
        sentences_tokens = []
        sentence, sentence_postags, sentence_labels = [],[], []
        tokens = []
        
        for l in lines:
            if l != []:
                
                if l[0] in ["-EOS-","-BOS-"]:
                    tokens.append(l[0])
                    sentence_postags.append(l[-2]) 
                else:     
                    tokens.append(l[0])
                    sentence.append(self._preprocess(l[0]))
                    sentence_labels.append(l[-1].strip())       
                    sentence_postags.append(l[-2]) 
            else:
                
                sentences_texts.append(" ".join(sentence))
                sentences_labels.append(sentence_labels)
                sentences_postags.append(sentence_postags)
                sentences_tokens.append(tokens)
                sentence, sentence_postags, sentence_labels = [], [] ,[]
                tokens = []

        assert(len(sentences_labels), len(sentences_texts))
        assert(len(sentence_postags), len(sentences_texts))
        for guid, (sent, labels) in enumerate(zip(sentences_texts, sentences_labels)):
 
            examples.append(
                InputSLExample(guid=guid, text_a=sent,
                               text_a_list=sentences_tokens[guid],
                               text_a_postags=sentences_postags[guid], 
                               labels=labels))
        return examples




def _valid_wordpiece_indexes(sent, wp_sent): 
    
    valid_idxs = []
    missing_chars = ""
    idx = 0
    
    for wp_idx, wp in enumerate(wp_sent,0):
        if sent[idx].startswith(wp) and missing_chars == "":
            valid_idxs.append(wp_idx)
        if missing_chars == "":
            missing_chars = sent[idx][len(wp.replace("##","")):]
        else:
            missing_chars = missing_chars[len(wp.replace("##","")):]
        
        if missing_chars == "":
            idx+=1
        
    return valid_idxs
    



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    label_map_reverse = {i:label for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        ori_tokens_a = example.text_a.split(" ")
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        ori_tokens_a =  ["[CLS]"] + ori_tokens_a + ["[SEP]"]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        valid_indexes = _valid_wordpiece_indexes(ori_tokens_a, tokens)
        input_mask = [1 if idtoken in valid_indexes else 0 
                      for idtoken, _ in enumerate(tokens)]

        labels_ids = []
        i=0
        for idtoken, token in enumerate(tokens):
            if idtoken in valid_indexes:
                
                if token == "[CLS]":
                    labels_ids.append(label_map["-BOS-"])
                elif token == "[SEP]":
                    labels_ids.append(label_map["-EOS-"])
                else:
                    try:
                        labels_ids.append(label_map[example.labels[i]])
                    except KeyError:
                        labels_ids.append(0)
                    i+=1
            else:        
                try:        
                    labels_ids.append(label_map[example.labels[min(i, len(example.labels)-1)]])
                except KeyError:
                    labels_ids.append(0)
                
        padding = [0] * (max_seq_length - len(input_ids))
        
        
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        labels_ids += padding

#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1] * len(input_ids)
#         # Zero-pad up to the sequence length.
#         padding = [0] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         input_mask += padding
#         segment_ids += padding    
#         labels_ids = [label_map[label] for label in example.labels]# label_map[example.labels]
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels_ids) == max_seq_length
        

#         if ex_index < 5:
#             logger.info("*** Example ***")
#             logger.info("guid: %s" % (example.guid))
#             logger.info("tokens: %s" % " ".join(
#                     [str(x) for x in tokens]))
#             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#             logger.info(
#                     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #logger.info("label: %s (id = %d)" % (example.labels, labels_ids))

        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              labels_ids=labels_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels, mask):

    output = out*mask
    gold = labels*mask
    
    
    mask = list()
    o_flat = list(output.flatten())
    g_flat = list(gold.flatten())


    o_filtered, g_filtered = [], []
    
    for o,g in zip(o_flat,g_flat):
        if g !=0:
            g_filtered.append(g)
            o_filtered.append(o)

    assert(len(o_filtered), len(g_filtered))
    return accuracy_score(o_filtered, g_filtered)
    

    
def posprocess_labels(preds):
    
    #This situation barely happens with LSTM's models
    for i in range(1, len(preds)-2):
        if preds[i] in ["-BOS-","-EOS-"] or preds[i].startswith("NONE"):
            preds[i] = "1ROOT@S"
            
    if len(preds) != 3 and not preds[-2].startswith("NONE"): preds[-2] = "NONE"
    if preds[-1] != "-EOS-": preds[-1] = "-EOS-"    
    
    if len(preds)==3 and preds[1] == "ROOT":
        preds[1] = "NONE"        
    
    return preds



def evaluate(model, device, logger, processor,data_dir, max_seq_length, tokenizer, label_list, 
             eval_batch_size, output_dir, 
             #path_evaluation_script, 
             path_gold, 
             #path_x2labels, 
             test, parsing_paradigm, evaluation_params = False):    
    

    if test:
        eval_examples = processor.get_test_examples(data_dir)
    else:
        eval_examples = processor.get_dev_examples(data_dir)
    
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    
    label_map_reverse = {i:l for i,l in enumerate(label_list)}
    
    examples_texts = [example.text_a_list for example in eval_examples]
    examples_postags = [example.text_a_postags for example in eval_examples]
    examples_preds = []
        
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
    
        with torch.no_grad():
            
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
    
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        masks = input_mask.cpu().numpy()
        outputs = np.argmax(logits, axis=2)
    
        for prediction, mask in zip(outputs, masks):
            examples_preds.append([label_map_reverse[element] for element, m in zip(prediction, mask)
                                   if m != 0])
            
        for idx_out, (o, l) in enumerate(zip(outputs,label_ids)):
            eval_accuracy += accuracy(o, l, masks[idx_out])
    
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    
    
    output_file_name = output_dir+".dev.outputs.txt.seq_lu" if not test else output_dir+".test.outputs.txt.seq_lu"
    with open(output_file_name,"w") as temp_out:
    #with tempfile.NamedTemporaryFile("w", delete=False) as temp_out:    
        content = []
        for tokens, postags, preds in zip(examples_texts, examples_postags, examples_preds):
            content.append("\n".join(["\t".join(element) for element in zip(tokens, postags, preds)]))
        temp_out.write("\n\n".join(content))
        temp_out.write("\n\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}
    
    output_eval_file = os.path.join(output_dir.rsplit("/",1)[0], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if parsing_paradigm.lower() == "dependencies":

        command = [#"PYTHONPATH="+abspath(join(dirname(__file__), data.dep2labels)),
                           "python",
                           "./dep2labels/decode_output_file.py",
                           #path_x2labels+os.sep+"decode_output_file.py", 
                           "--input", 
                           temp_out.name,
                           "--output",
                           temp_out.name.replace(".seq_lu","")+".conllu"
                      ]
                
        p = subprocess.Popen(" ".join(command),stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()

        #options = "--verbose" if test else ""
        options = "--verbose"
        command = ["python",
                   "./dep2labels/conll17_ud_eval.py",
                   #path_evaluation_script,#  
                   path_gold,
                   temp_out.name.replace(".seq_lu","")+".conllu", 
                   #temp_out.name+".out"
                   options]
    
        p = subprocess.Popen(" ".join(command),stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        score = [l for l in out.decode("utf-8").split("\n")
               if l.startswith("LAS")]
        out = out.decode("utf-8")
        score = float(score[0].strip().split("|")[3])

#         if test:
#             print (out.decode("utf-8"))

    elif parsing_paradigm.lower() == "constituency":
    
        sentences = [[(line.split("\t")[0],line.split("\t")[1]) for line in sentence.split("\n")] 
                        for sentence in content#.split("\n\n")
                        if sentence != ""]
    
        preds = [posprocess_labels([line.split("\t")[-1] for line in sentence.split("\n")]) 
                         for sentence in content#.split("\n\n")
                         if sentence != ""]

        parenthesized_trees = sequence_to_parenthesis(sentences,preds)#,None,None,None)            
    
        output_file_name = output_dir+".dev.outputs.txt" if not test else output_dir+".test.outputs.txt"
        with open(output_file_name,"w") as f_out:
            f_out.write("\n".join(parenthesized_trees))
    
        #with open(output_dir+os.se,"w") as temp_evalb:
        with tempfile.NamedTemporaryFile("w",delete=False) as temp_evalb: 
            command = [#"PYTHONPATH="+path_x2labels,
                       "python",
                       #path_x2labels+"/evaluate.py",
                       "./tree2labels/evaluate.py",
                       " --input ",
                       temp_out.name," --gold ",
                       path_gold,
                       " --evalb ", './tree2labels/EVALB/evalb']
                       #path_evaluation_script]
            
            if evaluation_params:
            #if path_evaluation_params is not None:
            
                #command.extend(["--evalb_param", path_evaluation_params])#,">",temp_evalb.name]
                command.extend(["--evalb_param", './tree2labels/EVALB/COLLINS.prm'])
            p = subprocess.Popen(" ".join(command),stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            out = out.decode("utf-8")
        score = float([l for l in out.split("\n")
                        if l.startswith("Bracketing FMeasure")][0].split("=")[1])
    else:
        raise NotImplementedError("Unknown parsing paradigm")
    return eval_loss, eval_accuracy, score, out
    


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory model will be written.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                      #  required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action="store_true",
                        help="Whether to run eval on the test set")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    parser.add_argument("--not_finetune", dest="not_finetune", default=False, action="store_true",
                        help="Determine where to finetune BERT (flag to True) or just the output layer (flag set to False)")
    
    parser.add_argument("--use_bilstms",
                        default=False,
                        action="store_true",
                        help="Further contextualized BERT outputs with BILSTMs")
    
    
    parser.add_argument("--evalb_param",
                       help="[True|False] to indicate whether use the COLLINS.prm parameter file")
    
    parser.add_argument("--parsing_paradigm",
                        type=str,
                        help="[constituency|dependencies]")
    
    parser.add_argument("--path_gold_parenthesized",
                        type=str,
                        help="Path to the constituency parenthesized files against which to compute the EVALB script")


    parser.add_argument("--path_gold_conllu",
                        type=str,
                        help="Path to the gold file in conllu formart")      

    args = parser.parse_args()


    processors = {"sl_tsv": SLProcessor}


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

#     if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
#         raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
#     os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = len(processor.get_labels(args.data_dir)) #num_labels_task[task_name]
    label_list = processor.get_labels(args.data_dir)
    label_reverse_map = {i:label for i, label in enumerate(label_list)}
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)


    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()



    model = MyBertForTokenClassification.from_pretrained(args.bert_model,
                                                       cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                       num_labels=num_labels,
                                                       finetune=not args.not_finetune,
                                                       use_bilstms=args.use_bilstms)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)            
        all_label_ids = torch.tensor([f.labels_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_dev_evalb = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
            
            if args.parsing_paradigm.lower() == "constituency":
                path_gold = args.path_gold_parenthesized
                evaluation_params = True if args.evalb_param is not None and args.evalb_param.lower() == "true" else False
            elif args.parsing_paradigm.lower() == "dependencies":
                path_gold = args.path_gold_conllu
                evaluation_params = False
            else:
                raise NotImplementedError("Unknown parsing paradigm")
            
            dev_loss, dev_acc, dev_eval_score, _ = evaluate(model, device, logger, processor, args.data_dir, 
                                                   args.max_seq_length, tokenizer, label_list,
                                                   args.eval_batch_size, args.model_dir, 
                                                   #path_evaluation,
                                                    path_gold,
                                                   #path_x2labels,
                                                   False,
                                                   parsing_paradigm=args.parsing_paradigm.lower(),
                                                   evaluation_params=evaluation_params)
            
            print ("Current best on the dev set: ", best_dev_evalb)
            if args.parsing_paradigm.lower() == "constituency":
                print ("Using evaluation params file:", evaluation_params)
            if best_dev_evalb < dev_eval_score:
                print ("New best on the dev set: ", dev_eval_score)
                best_dev_evalb = dev_eval_score     
                # Save a trained model
        
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.model_dir)
              #  output_model_file = os.path.join(args.model_dir, "pytorch_model.bin")
                
                if args.do_train:
                    print ("Saving the best new model...")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    
            model.train() #If not, following error: cudnn RNN backward can only be called in training mode

    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.model_dir)
    model_state_dict = torch.load(output_model_file)

    model = MyBertForTokenClassification.from_pretrained(args.bert_model,
                                                       state_dict=model_state_dict,
#                                                       cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                       num_labels=num_labels,
                                                       finetune=not args.not_finetune,
                                                       use_bilstms=args.use_bilstms)    

    model.to(device)

    if (args.do_eval or args.do_test) and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        if args.parsing_paradigm.lower() == "constituency":
            evaluation_params = True if args.evalb_param is not None and args.evalb_param.lower() == "true" else False 
            path_gold = args.path_gold_parenthesized
        elif args.parsing_paradigm.lower() == "dependencies":
            path_gold = args.path_gold_conllu
            evaluation_params=False
        else:
            raise NotImplementedError("Unknown parsing paradigm")
            
        loss, acc, eval_score, detailed_score = evaluate(model, device, logger, processor, args.data_dir, 
                                               args.max_seq_length, tokenizer, label_list,
                                               args.eval_batch_size, 
                                               args.output_dir,  
                                               path_gold,
                                               args.do_test,
                                               parsing_paradigm=args.parsing_paradigm.lower(),
                                               evaluation_params=evaluation_params)
            
        print (detailed_score)
if __name__ == "__main__":
    main()
