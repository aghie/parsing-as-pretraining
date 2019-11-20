"""
Takes an output following the tree2labels encoding, transforms it into
a parenthesized tree and evaluates it following the EVALB script

python /home/david/git/tree2labels/evaluate.py \
--input /tmp/ptb-dev.reversed \
--gold /home/david/Escritorio/PTB_pred_tags/dev.trees \
--evalb /home/david/git/tree2labels/EVALB/evalb

"""

from argparse import ArgumentParser
from utils import sequence_to_parenthesis, rebuild_input_sentence
from subprocess import PIPE,Popen
import codecs
import os
import sys
import uuid


#TODO: Can be done better? Does not consider a valid label for the CTB. Almost never happens, just used as a security check
def posprocess_labels(preds):
    
  #  print ("PREDS", preds)
    #This situation barely happens with LSTM's models
    for i in range(1, len(preds)-2):
        if "-BOS-" in preds[i] or "-EOS-" in preds[i] or preds[i].startswith("NONE"):
            preds[i] = "1ROOT@S"
    
    if not preds[-2].startswith("NONE") and len(preds) != 3: 
        preds[-2] = "NONE"
    if preds[-1] != "-EOS-": preds[-1] = "-EOS-"    

    #TODO: This is currently needed as a workaround for the retagging strategy and sentences of length one
    if len(preds)==3 and preds[1] == "ROOT":
        preds[1] = "NONE"        
    
    #print ("PREDS AFTER", preds)
    return preds

if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", dest="input", 
                            help="Path to the original encoding used in Constituent Parsing as Sequence Labeling", 
                            default=None)
    arg_parser.add_argument("--gold", dest="gold",
                            help="Path to the parenthesized (gold) version of the same trees as in --input")
    arg_parser.add_argument("--join_char", dest="join_char", default="~",
                            help="Used symbol to collapse/uncollapse unary branches")
    arg_parser.add_argument("--split_char", dest="split_char", default="@",
                            help="Used symbol to concatenate the partial labels for a word")
    arg_parser.add_argument("--evalb", dest="evalb",
                            help="Path to the EVALB script")
    arg_parser.add_argument("--evalb_param", dest="evalb_param",
                            help="Path to the EVALB param file", default=None)
    
    args = arg_parser.parse_args()
    
   # reload(sys)
   # sys.setdefaultencoding('UTF8')

    gold_trees = codecs.open(args.gold).readlines()
    
    #Check if we need to add an a pair of ROOT brackets (needed for SPRML)?
    add_root_brackets = False
    if gold_trees[0].startswith("( ("):
        add_root_brackets = True
    
    with codecs.open(args.input) as f_input:
        raw_sentences = f_input.read().split("\n\n")
        all_sentences = []
        all_preds = []
        for raw_sentence in raw_sentences:

            lines = raw_sentence.split("\n")
       #     print ("raw lines", lines)
            if len(lines) != 1:
                
                #l.split("\t")[1:-1]
                
                
                #if len(l.split("\t")) > 3:
                
                sentence = rebuild_input_sentence(lines)
                
#                 if len(lines[0].split("\t")) > 3:
#                     
#                     sentence = [ (l.split("\t")[0],  
#                                       l.split("\t")[1]+"##"+"|".join(feat for feat in l.split("\t")[2:-1]
#                                                if feat != "-")+"##") 
#                                     
#                                 for l in lines]                    
#                 else:
#                     sentence = [tuple(l.split("\t")[0:2]) for l in lines]

                
        #        print "sentence",sentence
                
                #sentence = [tuple(l.split("\t")[0:2]) for l in lines]
                #Posprocessing should be only needed for the first epochs, were some invalid trees might still occur
                preds_sentence = posprocess_labels([l.split("\t")[-1] for l in lines])
                
        #        print "preds_sentence", preds_sentence

        #        exit()
         #       raw_input("NEXT")
                
                all_sentences.append(sentence)
                all_preds.append(preds_sentence)
        
        
        parenthesized_trees = sequence_to_parenthesis(all_sentences,all_preds, 
                                                      join_char=args.join_char, split_char=args.split_char)#,None,None,None)
        
        
        if add_root_brackets:
            parenthesized_trees = ["( "+line+")" for line in parenthesized_trees]
        
        
        
 #       print all_preds[0]
#         print (parenthesized_trees[0])
#         print (gold_trees[0])
#         print ()
#         print (parenthesized_trees[1])
#         print (gold_trees[1])
      #  exit()
        
        file_name = "/tmp/"+str(uuid.uuid4())
        f = codecs.open(file_name,"w")
        for tree in parenthesized_trees:
            f.write(tree)
            f.write("\n")

     #   print (file_name)
     #   exit()

        if args.evalb_param is None:
            command = [args.evalb,args.gold, f.name]
        else:
            command = [args.evalb,"-p",args.evalb_param,args.gold,f.name]
#        p = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#        output, err = p.communicate()  
#        print output
        os.system(" ".join(command))
        os.remove(f.name)
