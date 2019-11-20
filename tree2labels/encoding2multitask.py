from argparse import ArgumentParser
from utils import sequence_to_parenthesis, flat_list, rebuild_input_sentence
from tree import SeqTree, SyntacticDistanceEncoder
from collections import Counter
import codecs
import os
import copy
import sys
import warnings

"""
To encode:

python /home/david/Escritorio/encoding2multitask.py \
--input /home/david/Escritorio/dataset/ptb/ptb-dev.seq_lu \
--output /tmp/ptb-dev.multitask \
--status encode

To decode:

python /home/david/Escritorio/encoding2multitask.py \
--input /tmp/ptb-test.multitask \
--output /tmp/ptb-test.reversed \
--status decode

"""



def tag_to_multitask(tag, multitask_char, split_char):
    tag_split = tag.split(split_char)
    
    #It is a tag that encodes (level, label, leaf unary branch)
    if len(tag_split) == 3:
        return multitask_char.join(tag_split)
    #It is a regular tag
    elif len(tag_split) == 2:
        return multitask_char.join((tag_split[0], tag_split[1], "-EMPTY-"))
    elif tag in ["-BOS-","-EOS-", "NONE"]:
        return multitask_char.join([tag,tag,tag])
    else:
        warnings.warn("The expected multitask label only contains labels for one task: "+tag)
        return multitask_char.join([tag,"-EMPTY-","-EMPTY-"])
        
#        print tag, tag_split
#        raise NotImplementedError("len(tag_split)==1")


def is_beginning_of_chunk(c,c_before):
    return c != c_before and c in ["NP","VP", "PP"] 

def is_end_of_chunk(c,c_next):
    return c != c_next

def is_chunkeable(c):
    return c in ["NP","VP","PP"]


def unk_word(word, most_common, uncommon):
    
    if word == "-EMPTY-":
        return word
    
    if word.isdigit():
        return "0"

    if word not in uncommon:
        return "-oov-"

    if word not in most_common:
        return "-unk-"
    
    
    

    
    return word
    
    


"""
Returns a sequence of labels to predict the position located
at n positions to the right
"""
def to_next_label(labels,n):
    
    next_labels = []
    for idl,l in enumerate(labels):
        
        if n > 0:
        
            if idl+n > len(labels)-n:
                next_labels.append("-EMPTY-")
            else:
                next_labels.append(labels[idl+n])
                
        else:
            
            if idl+n < 0:
                next_labels.append("-EMPTY-")
            else:
                next_labels.append(labels[idl+n])
                
    return next_labels

"""
NOTE: This is not really useful in the current format
"""
def to_chunks(constituents):
    
    
    chunk_sequence = []
    c_before = None
    for idc,c in enumerate(constituents):
        
        if is_beginning_of_chunk(c, c_before):
            chunk_sequence.append("B-"+c)
        elif is_chunkeable(c):
            chunk_sequence.append("I-"+c)
        else:
            chunk_sequence.append("O")
        
        c_before = c
    return chunk_sequence
        


"""
Transforms an encoding of a tree in a relative scale into an
encoding of the tree in an absolute scale.
"""
def to_absolute_levels(relative_levels, levels_to_encode):
    
    absolute_sequence = [0]*len(relative_levels)
    current_level = 0
    for j,level in enumerate(relative_levels):
    
        if level in ["-BOS-","-EOS-", "NONE"]:
            absolute_sequence[j] = "-EMPTY-"
        else:
            
            if level == "ROOT":
                current_level=1
                label_j=str(current_level)
            
            elif "ROOT" in level:
                current_level+=int(level.replace("ROOT",""))
                label_j = str(current_level)
            else:                
                current_level+= int(level)
                label_j=str(current_level)
            
            if int(label_j) <= levels_to_encode:
                absolute_sequence[j] = label_j
            else:
                absolute_sequence[j] = "-EMPTY-"
                
    return absolute_sequence


    

#TODO: What to do if not for all tasks we return a -BOS-/-EOS- when needed. Voting approach?
def multitag_to_tag(multitag, multitask_char, split_char):
    
    multitag_split = multitag.split(multitask_char)[0:3]
    
    if multitag_split[1] in ["-BOS-","-EOS-","NONE"]:
        return multitag_split[1]
    
    if multitag_split[2] != "-EMPTY-":
        return split_char.join(multitag_split)
    else:
        return split_char.join(multitag_split[0:2])
    
    
def add_tag(l, new_elements, multitask_char):
    
    for idtoken, token in enumerate(l):
        token[2] += multitask_char+new_elements[idtoken]

def decode_int(preds):
    #f_output = codecs.open(args.output,"w")
    decoded_output = ''
    sentence = []
    #with codecs.open(args.input) as f_input:
    #lines = f_input.readlines()
#    print(preds)
    for l in preds.split('^^'):
        if l != "\n":
#            print(l)
            word,postag,label = l.strip().split("\t")
            label = multitag_to_tag(label,"{}","@") #The tasks that we care about are just the first three ones.
            sentence.append([word,postag,label])
            #f_output.write("\t".join([word,postag,label])+"\n")
        else:
#            print("END")
            for token in sentence:
                decoded_output += "\t".join(token)+"\n"
                #f_output.write("\t".join(token)+"\n")
            sentence = []
            #f_output.write("\n")
            decoded_output +="\n"
#    print("dec: ",decoded_output)
    return decoded_output   

if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", dest="input", 
                            help="Path to the original encoding used in Constituent Parsing as Sequence Labeling", 
                            default=None)
    arg_parser.add_argument("--output", dest="output", 
                            help="Path to the output encoding, formatted as multitask learning", default=None)
    arg_parser.add_argument("--status", dest="status",
                            help="[encode|decode]")
    arg_parser.add_argument("--add_abs_scale", dest="add_abs_scale", action="store_true", default=False,
                            help="Add the prediction of the level in absolute scale as an auxiliary tasks")
    arg_parser.add_argument("--abs_levels",dest="abs_levels",
                            help="Levels for which to predict the absolute scale. An integer number")
    arg_parser.add_argument("--add_chunks", dest="add_chunks",action="store_true",
                            help="Add chunks as an auxiliary task")
    arg_parser.add_argument("--add_next_level", dest="add_next_level", action="store_true",
                            help="Add the prediction of the next level as an auxiliary task")
    arg_parser.add_argument("--add_prev_level", dest="add_prev_level", action="store_true",
                            help="Ad the prediction of the previous level as an auxiliary task")
    arg_parser.add_argument("--add_next_label", dest="add_next_label", action="store_true",
                            help="Add the prediction of the next label as an auxiliary task")
    arg_parser.add_argument("--add_prev_label", dest="add_prev_label", action="store_true",
                            help="Add the prediction of the prev label as an auxiliary task")
    arg_parser.add_argument("--add_syntactic_distances", dest="add_syntactic_distances", action="store_true",
                            help="Add the prediction of syntactic distances as an auxiliary task")
    arg_parser.add_argument("--add_next_word", dest="add_next_word", action="store_true",
                            help="Add the prediction of the next word as an auxiliary task")
    arg_parser.add_argument("--add_prev_word", dest="add_prev_word", action="store_true",
                            help="Add the prediction of the prev word as an auxiliary task")
    arg_parser.add_argument("--common_words", dest="common_words",
                            help="Path to the file containing the list of common words")
    arg_parser.add_argument("--uncommon_words", dest="uncommon_words",
                            help="Path to th file containing the list of uncommon words")
    arg_parser.add_argument("--split_char", dest="split_char",type=str,
                            default="@")
    arg_parser.add_argument("--multitask_char", dest="multitask_char",type=str,
                            default="{}")    
    

    args = arg_parser.parse_args()
    
    auxiliary_tasks = [] #["absolute_scale"]
    sentence = []
    
    reload(sys)
    sys.setdefaultencoding('UTF8')

    if args.status == "encode":
    
        f_dest = codecs.open(args.output,"w")
        
        with codecs.open(args.input) as f_input:
            lines = f_input.readlines()

        if args.add_next_word or args.add_prev_word:

            with codecs.open(args.common_words) as f:
                most_common = set([l.strip("\n") for l in f.readlines()])
                
            with codecs.open(args.uncommon_words) as f:
                uncommon = set([l.strip("\n") for l in f.readlines()])
        

        #Compute number of words, postags and labels for 
        #different purposes
            
        relative_levels = []
        label_sequences = []
        ori_input, ori_labels = [],[]
        words = []

        for l in lines:
            if l != "\n":
                word,postag,label = l.strip().split("\t")[0], "\t".join(l.strip().split("\t")[1:-1]), l.strip().split("\t")[-1]
                #tuple(l.strip().split("\t"))
                #word,postag,label = tuple(l.strip().split("\t"))
                words.append(word)
                
                if args.add_syntactic_distances:
                    ori_input.append((word,postag))
                    ori_labels.append(label)
                
                label = tag_to_multitask(label, args.multitask_char, args.split_char)
                
                if args.add_abs_scale or args.add_next_level or args.add_prev_level:
                    relative_levels.append(label.split(args.multitask_char)[0])
                
                if args.add_chunks or args.add_next_label or args.add_prev_label:
                    label_sequences.append(label.split(args.multitask_char)[1])    
                
                sentence.append([word,postag,label])
                #f_output.write("\t".join([word,postag,label])+"\n")
            else:
                if args.add_abs_scale:
                     absolute_levels = to_absolute_levels(relative_levels, int(args.abs_levels))
                     add_tag(sentence,absolute_levels,args.multitask_char)

                
                if args.add_chunks:
                     chunks = to_chunks(label_sequences)
                     add_tag(sentence,chunks,args.multitask_char)
       
                
                #Predicting the next and the previous levels
                if args.add_next_level:
                     next_levels = to_next_label(relative_levels, 1)
                     add_tag(sentence,next_levels,args.multitask_char)

                if args.add_prev_level:
                    prev_levels = to_next_label(relative_levels, -1)
                    add_tag(sentence, prev_levels,args.multitask_char)
                    
                #Predicting the next and the previous labels
                if args.add_next_label:
                    next_labels = to_next_label(label_sequences,1)
                    add_tag(sentence, next_labels,args.multitask_char)
                    
                if args.add_prev_label:
                    prev_labels = to_next_label(label_sequences,-1)
                    add_tag(sentence, prev_labels,args.multitask_char)

                #Predicting the next and previous word
                if args.add_next_word:
                    next_words  = to_next_label(words, 1)
                    aux = []
                    for w in next_words:
                        aux.append(unk_word(w,most_common,uncommon))
                    add_tag(sentence,aux,args.multitask_char)
         
                if args.add_prev_word:
                    prev_words = to_next_label(words,-1)
                    aux = []
                    for w in prev_words:
                        aux.append(unk_word(w,most_common,uncommon))
                    add_tag(sentence, aux,args.multitask_char)
                
                if args.add_syntactic_distances:

                    tree = sequence_to_parenthesis([ori_input], [ori_labels])
                    tree = SeqTree.fromstring(tree[0], remove_empty_top_bracketing=True)
                    
                    
                    tree.collapse_unary(collapsePOS=True, collapseRoot=True)    
                    syntactic_distances = []
                    SyntacticDistanceEncoder().encode(tree, syntactic_distances)
                    
                    #reversing the labels
                    set_distances = set(syntactic_distances)
                    set_distances.remove(0)
                    ori_order = sorted(set_distances, reverse=True)
                    reverse_order = sorted(set_distances, reverse=False)
                    
#If we want to do it reversed, but then it is closer to our encoding and pottentially not so useful 
#                     map_distances = {o:r for o, r in zip(ori_order, reverse_order)}
#                     map_distances.update({0:0})
#                     reversed_syntactic_distances = ["-1"]
#                     reversed_syntactic_distances.extend([str(map_distances[d]) for d in syntactic_distances])
#                     reversed_syntactic_distances.append("-1")
                    
                    syntactic_distances.insert(0,"-1")
                    syntactic_distances.append("-1")
                    add_tag(sentence, [str(s) for s in syntactic_distances],args.multitask_char)
                
                #sentence.append([""])
                for token in sentence:
                    f_dest.write("\t".join(token)+"\n")
                
                f_dest.write("\n")

                sentence = []
                relative_levels = []
                absolute_levels = []
                label_sequences = []
                ori_input = []
                ori_labels = []
                words = []
                
    elif args.status == "decode":

        f_output = codecs.open(args.output,"w")
        labels = []
        with codecs.open(args.input) as f_input:
            lines = f_input.readlines()
        
        for l in lines:
            if l != "\n":
                word,postag,label = l.strip().split("\t")[0], "\t".join(l.strip().split("\t")[1:-1]), l.strip().split("\t")[-1]
                #word,postag,label = l.strip().split("\t")
                label = multitag_to_tag(label, args.multitask_char, args.split_char) #The tasks that we care about are just the first three ones.
                sentence.append(l)
                labels.append(label)
                #sentence.append([word,postag,label])
                #f_output.write("\t".join([word,postag,label])+"\n")
            else:
                for token,label in zip(rebuild_input_sentence(sentence), labels):
                    f_output.write("\t".join(token)+"\t"+label+"\n")
                sentence = []
                labels = []
                f_output.write("\n")
        
        
        
