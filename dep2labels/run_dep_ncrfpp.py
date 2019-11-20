'''

'''

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from decodeDependencies import decode
import codecs
import os
import tempfile
import copy
import time
import sys
import uuid
STATUS_TEST = "test"
STATUS_TRAIN = "train"


if __name__ == '__main__':
     
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--test", dest="test",help="Path to the input test file as sequences", default=None)
    arg_parser.add_argument("--gold", dest="gold", help="Path to the original linearized trees, without preprocessing")
    arg_parser.add_argument("--model", dest="model", help="Path to the model")
    arg_parser.add_argument("--status", dest="status", help="[train|test]")
    arg_parser.add_argument("--gpu",dest="gpu",default="False")
    arg_parser.add_argument("--output",dest="output",default="/tmp/trees.txt", required=True)
    arg_parser.add_argument("--ncrfpp", dest="ncrfpp", help="Path to the NCRFpp repository")
    arg_parser.add_argument("--conll_ud", dest="conll_ud", help="Path to the conll ud evaluation script")
    
    args = arg_parser.parse_args()    
    
    if args.status.lower() == STATUS_TEST:
        
        #gold_trees = codecs.open(args.gold).readlines()
        path_raw_dir = args.test
        path_name = args.model
        path_output = "/tmp/"+path_name.split("/")[-1]+".output"
        path_tagger_log = "/tmp/"+path_name.split("/")[-1]+".tagger.log"
        path_dset = path_name+".dset"
        path_model = path_name+".model"
        
        #Reading stuff for evaluation
        sentences = []
        gold_labels = []
        for s in codecs.open(path_raw_dir).read().split("\n\n"):
            sentence = []
            for element in s.split("\n"):
                if element == "": break
                word,postag,label = element.strip().split("\t")[0], "\t".join(element.strip().split("\t")[1:-1]), element.strip().split("\t")[-1]
                #word,postag,label = element.split("\t")
                sentence.append((word,postag))
                gold_labels.append(label)
            if sentence != []: sentences.append(sentence)

        
        end_merge_retags_time = 0
        time_mt2st = 0 
        #If we follow the retagging approach, we first need to retag the corpus
        #This implies to execute a second classifier
 
        conf_str = """
        ### Decode ###
        status=decode
        """
        conf_str+="raw_dir="+path_raw_dir+"\n"
        conf_str+="decode_dir="+path_output+"\n"
        conf_str+="dset_dir="+path_dset+"\n"
        conf_str+="load_model_dir="+path_model+"\n"
        conf_str+="gpu="+args.gpu+"\n"

        decode_fid = str(uuid.uuid4())
        decode_conf_file = codecs.open("/tmp/"+decode_fid,"w")
        decode_conf_file.write(conf_str)

        os.system("python "+args.ncrfpp+"/main.py --config "+decode_conf_file.name +" > "+path_tagger_log)


        #We decode the multitask output. We measure the time, because actually could be saved
        #if done directly on the NCRFpp
        
        #We re-read the sentences in case we are applying the retagging approach and we need to load
        #the retags
        output_file = codecs.open(path_output)
        output_lines = output_file.readlines()
        output_file = codecs.open(path_output)
        output_content = output_file.read()
        
#         print (output_content)
#         
        sentences = [[(line.split("\t")[0],line.split("\t")[1]) for line in sentence.split("\n")] 
                     for sentence in output_content.split("\n\n")
                     if sentence != ""]
#         
#         print (sentences)
#         
#         #I updated the output of the NCRFpp to be a tsv file
#         init_posprocess_time = time.time()
        preds = [ [line.split("\t")[-1]
                                     for line in sentence.split("\n")]
                     for sentence in output_content.split("\n\n")
                     if sentence != ""]
 
    
     #   tmpfile = codecs.open(args.output,"w")
     
     #   print (output_content.readlines())
       # print (output_lines)
    #    print (output_content)
        decode(output_lines, args.output, "@", "English")
       # decode(output_file.readlines(), args.output, "@")


        #We read the time that it took to process the samples from the NCRF++ log file.
        log_lines = codecs.open(path_tagger_log).readlines()
        raw_time = float([l for l in log_lines
                    if l.startswith("raw: time:")][0].split(",")[0].replace("raw: time:","").replace("s",""))
        raw_unary_time = 0
        #If we applied the retagging strategy, we also need to consider the time that it took to execute the retagger
        
        command = " ".join(["python",args.conll_ud, args.gold, args.output, "--verbose"])
      
        os.system(command)   

        #os.system(" ".join([args.evalb,args.gold, tmpfile.name]))
        os.remove("/tmp/"+decode_fid)

      #  total_time = raw_time+raw_unary_time+end_posprocess_time+end_parenthesized_time+end_merge_retags_time
        #We do no need to count the time of this script as adapting the NCRFpp for this would be straight forward
     #   total_time-= time_mt2st
        
#         print ("Total time:",round(total_time,2)) 
#         print ("Sents/s:   ", round(len(gold_trees)/(total_time),2))
        
        #We are also saving the output in a sequence tagging format, in case we want to perform
        #a further analysis
        with codecs.open(args.output+".seq_lu","w") as f_out_seq_lu:
            for (sentence, sentence_preds) in zip(sentences,preds):
                for ((w, pos), l) in zip(sentence, sentence_preds):
                    f_out_seq_lu.write("\t".join([w,pos,l])+"\n")
                f_out_seq_lu.write("\n")
    
      #NEEDs to be adapted with the multitask setup
      #  print "Accuracy:  ",  round(accuracy_score(gold_labels, flat_preds),4)


        
