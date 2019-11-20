from argparse import ArgumentParser
import codecs
import os

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


def tag_to_multitask(tag):
    tag_split = tag.split("_")
    
    #It is a tag that encodes (level, label, leaf unary branch)
    if len(tag_split) == 3:
        return "{}".join(tag_split)
    #It is a regular tag
    elif len(tag_split) == 2:
        return "{}".join((tag_split[0], tag_split[1], "-EMPTY-"))
    elif tag in ["-BOS-","-EOS-", "NONE"]:
        return "{}".join([tag,tag,tag])
    else:
        raise NotImplementedError("len(tag_split)==1")



"""
Transforms an encoding of a tree in a relative scale into an
encoding of the tree in an absolute scale.
"""
def to_absolute_levels(relative_levels):
    
    absolute_sequence = [0]*len(relative_levels)
    current_level = 0
    for j,level in enumerate(relative_levels):
    
        if level in ["-BOS-","-EOS-", "NONE"]:
            absolute_sequence[j] = level
        elif level == "ROOT":
            absolute_sequence[j] = "1"
            current_level+=1
        else:                
            current_level+= int(level)
            absolute_sequence[j] = str(current_level)
    return absolute_sequence
    

#TODO: What to do if not for all tasks we return a -BOS-/-EOS- when needed. Voting approach?
def multitag_to_tag(multitag):
    
    multitag_split = multitag.split("{}")[0:3]
    
    if multitag_split[1] in ["-BOS-","-EOS-","NONE"]:
        return multitag_split[1]
    
    if multitag_split[2] != "-EMPTY-":
        return "_".join(multitag_split)
    else:
        return "_".join(multitag_split[0:2])

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
            label = multitag_to_tag(label) #The tasks that we care about are just the first three ones.
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
    arg_parser.add_argument("--split_char", dest="split_char",type=str,
                            default="@")
    arg_parser.add_argument("--multitask_char", dest="multitask_char",type=str,
                            default="{}")    
    
    args = arg_parser.parse_args()
    
    auxiliary_tasks = ["absolute_scale"]
    sentence = []

    if args.status == "encode":
    
        f_output = codecs.open(args.output,"w")
        
        with codecs.open(args.input) as f_input:
            lines = f_input.readlines()
        
        relative_levels = []
        for l in lines:
            if l != "\n":
                word,postag,label = l.strip().split("\t")
                label = tag_to_multitask(label)
                
                if "absolute_scale" in auxiliary_tasks:
                    relative_levels.append(label.split(args.multitask_char)[0])

                sentence.append([word,postag,label])
                #f_output.write("\t".join([word,postag,label])+"\n")
            else:

                if "absolute_scale" in auxiliary_tasks:
                    absolute_levels = to_absolute_levels(relative_levels)
                for idtoken, token in enumerate(sentence):
                    token[2] += "{}"+absolute_levels[idtoken]
                    f_output.write("\t".join(token)+"\n")
                f_output.write("\n")

                sentence = []
                relative_levels = []
                absolute_levels = []
                
    elif args.status == "decode":
        
        f_output = codecs.open(args.output,"w")
        
        with codecs.open(args.input) as f_input:
            lines = f_input.readlines()
        
        for l in lines:
            if l != "\n":
                word,postag,label = l.strip().split("\t")
                label = multitag_to_tag(label) #The tasks that we care about are just the first three ones.
                sentence.append([word,postag,label])
                #f_output.write("\t".join([word,postag,label])+"\n")
            else:
                for token in sentence:
                    f_output.write("\t".join(token)+"\n")
                sentence = []
                f_output.write("\n")
        
        
        
