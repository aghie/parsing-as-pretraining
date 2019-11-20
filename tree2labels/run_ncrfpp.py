from tree import SeqTree, RelativeLevelTreeEncoder
from argparse import ArgumentParser
from nltk.tree import Tree
from utils import sequence_to_parenthesis, get_enriched_labels_for_retagger, flat_list,rebuild_input_sentence
from sklearn.metrics import accuracy_score
import codecs
import os
import tempfile
import copy
import time
import sys
import uuid
STATUS_TEST = "test"
STATUS_TRAIN = "train"


def posprocess_labels(preds):
    
    #This situation barely happens with LSTM's models
    for i in range(1, len(preds)-2):
        if preds[i] in ["-BOS-","-EOS-"] or preds[i].startswith("NONE"):
            preds[i] = "1ROOT@S"
            
    if len(preds) != 3 and not preds[-2].startswith("NONE"): preds[-2] = "NONE"
    if preds[-1] != "-EOS-": preds[-1] = "-EOS-"    
    
    #TODO: This is currently needed as a workaround for the retagging strategy and sentences of length one
    if len(preds)==3 and preds[1] == "ROOT":
        preds[1] = "NONE"        
    
    return preds


if __name__ == '__main__':
     
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--test", dest="test",help="Path to the input test file as sequences", default=None)
    arg_parser.add_argument("--gold", dest="gold", help="Path to the original linearized trees, without preprocessing")
    arg_parser.add_argument("--model", dest="model", help="Path to the model")
    arg_parser.add_argument("--status", dest="status", help="[train|test]")
    arg_parser.add_argument("--retagger", dest="retagger", action="store_true", default=False)
    arg_parser.add_argument("--gpu",dest="gpu",default="False")
    arg_parser.add_argument("--multitask", dest="multitask", default=False, action="store_true")
    arg_parser.add_argument("--output",dest="output",default="/tmp/trees.txt", required=True)
    arg_parser.add_argument("--evalb",dest="evalb",help="Path to the script EVALB")
    arg_parser.add_argument("--evalb_param", dest="evalb_param", help="Path to the EVALB param file", default=None)
    arg_parser.add_argument("--ncrfpp", dest="ncrfpp", help="Path to the NCRFpp repository")
    
    args = arg_parser.parse_args()
    
    #If not, it gives problem with Chinese chracters
 #   reload(sys)
 #   sys.setdefaultencoding('UTF8')
    
    gold_trees = codecs.open(args.gold).readlines()
    
    #Check if we need to add an a pair of ROOT brackets (needed for SPRML)?
    add_root_brackets = False
    if gold_trees[0].startswith("( ("):
        add_root_brackets = True
    
    if args.status.lower() == STATUS_TEST:
        
        gold_trees = codecs.open(args.gold).readlines()
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

        
        unary_preds = None
        end_merge_retags_time = 0
        time_mt2st = 0 
        #If we follow the retagging approach, we first need to retag the corpus
        #This implies to execute a second classifier
        if args.retagger:
            
            path_raw_dir_unary = args.test
            path_output_unary = "/tmp/"+args.model.split("/")[-1]+"-unary.output"
            path_tagger_log_unary = "/tmp/"+args.model.split("/")[-1]+".tagger.unary.log"
            path_dset_unary = args.model+"-unary.dset"
            path_model_unary = args.model+"-unary.model"
            unary_conf_str = """
            ### Decode ###
            status=decode
            """
            
            unary_conf_str+="raw_dir="+path_raw_dir_unary+"\n"
            unary_conf_str+="decode_dir="+path_output_unary+"\n"
            unary_conf_str+="dset_dir="+path_dset_unary+"\n"
            unary_conf_str+="load_model_dir="+path_model_unary+"\n"
            unary_conf_str+="gpu="+args.gpu+"\n"

            #TODO: This should be done with NamedTemporaryFile, but having some problem with the Chinese files
            decode_unary_fid = str(uuid.uuid4())
            decode_unary_conf_file = codecs.open("/tmp/"+decode_unary_fid,"w")
            decode_unary_conf_file.write(unary_conf_str)        
            
            os.system("python "+args.ncrfpp+"/main.py --config "+decode_unary_conf_file.name+" > "+path_tagger_log_unary)
            #os.system("python /home/david.vilares/git/NCRFpp/main.py --config "+decode_unary_conf_file.name+" > "+path_tagger_log_unary)


            unary_preds = [[line.split(" ")[2] for line in sentence.split("\n")] 
                     for sentence in codecs.open(path_output_unary).read().split("\n\n")
                     if sentence != ""]
#Updated, now the output of the NCRFpp is a TSV file
#             unary_preds = [[line.split(" ")[2] for line in sentence.split("\n")] 
#                      for sentence in codecs.open(path_output_unary).read().split("\n\n")
#                      if sentence != ""]

            #We dump the retags into a file. From there they will be loaded to be parsed
            new_unary_preds = []
            fid = str(uuid.uuid4())
            f  = codecs.open("/tmp/"+fid,"w",encoding="utf-8")
            init_merge_retags_time = time.time()
            for j,sentence in enumerate(sentences):
                unary_pred = []
                for (word,postag), retag in zip(sentence,unary_preds[j]):
                        
                    if retag == "-EMPTY-" or word in ["-BOS-","-EOS-"]:
                        retag = postag
                    else:
                        retag = retag+"+"+postag
                            
                    unary_pred.append(retag)
                    f.write("\t".join([word,retag])+"\n")
                new_unary_preds.append(unary_pred)
                f.write("\n")
            end_merge_retags_time = time.time()-init_merge_retags_time
            path_raw_dir = f.name
            
            f.close()
        

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

        #print ("python "+args.ncrfpp+"/main.py --config "+decode_conf_file.name +" > "+path_tagger_log)
        #exit()
        os.system("python "+args.ncrfpp+"/main.py --config "+decode_conf_file.name +" > "+path_tagger_log)
        #os.system("python /home/david.vilares/git/NCRFpp/main.py --config "+decode_conf_file.name+" > "+path_tagger_log)

        #We decode the multitask output. We measure the time, because actually could be saved
        #if done directly on the NCRFpp
        
        if args.multitask:
            init_time_mt2st = time.time()
            os.system("python ../encoding2multitask.py --input "+path_output+" --output "+
                                path_output+"2st" +" --status decode")     
            path_output = path_output+"2st"
            time_mt2st = time.time() - init_time_mt2st 

        #We re-read the sentences in case we are applying the retagging approach and we need to load
        #the retags
        output_content = codecs.open(path_output).read()
        
        

        sentences = [rebuild_input_sentence(sentence.split("\n")) 
                     for sentence in output_content.split("\n\n")
                     if sentence != ""]        
        
        
#         sentences = [[(line.split("\t")[0],line.split("\t")[1]) for line in sentence.split("\n")] 
#                      for sentence in output_content.split("\n\n")
#                      if sentence != ""]
        
        
        #I updated the output of the NCRFpp to be a tsv file
        init_posprocess_time = time.time()
        preds = [ posprocess_labels([line.split("\t")[-1] if not args.multitask 
                                     else line.split("\t")[-1]
                                     for line in sentence.split("\n")]) 
                     for sentence in output_content.split("\n\n")
                     if sentence != ""]
#         preds = [ posprocess_labels([line.split("\t")[2] if not args.multitask 
#                                      else line.split("\t")[2]
#                                      for line in sentence.split("\n")]) 
#                      for sentence in output_content.split("\n\n")
#                      if sentence != ""]


        end_posprocess_time = time.time()-init_posprocess_time
        
        init_parenthesized_time = time.time()
        parenthesized_trees = sequence_to_parenthesis(sentences,preds)#,None,None,None)
        
        if add_root_brackets:
            parenthesized_trees = ["( "+line+")" for line in parenthesized_trees]
        
        end_parenthesized_time =  time.time()-init_parenthesized_time
        
        tmpfile = codecs.open(args.output,"w")
        tmpfile.write("\n".join(parenthesized_trees)+"\n")
        
        #We read the time that it took to process the samples from the NCRF++ log file.
        log_lines = codecs.open(path_tagger_log).readlines()
        raw_time = float([l for l in log_lines
                    if l.startswith("raw: time:")][0].split(",")[0].replace("raw: time:","").replace("s",""))
        raw_unary_time = 0
        #If we applied the retagging strategy, we also need to consider the time that it took to execute the retagger
        if args.retagger:
            log_lines_unary = codecs.open(path_tagger_log_unary).readlines()
            raw_unary_time = float([l for l in log_lines_unary
                        if l.startswith("raw: time:")][0].split(",")[0].replace("raw: time:","").replace("s",""))
            os.remove("/tmp/"+decode_unary_fid)
            os.remove("/tmp/"+fid)
        
        if args.evalb_param is None:
            os.system(" ".join([args.evalb, args.gold, tmpfile.name]))
        else:
            os.system(" ".join([args.evalb,"-p",args.evalb_param,args.gold, tmpfile.name]))
        os.remove("/tmp/"+decode_fid)    

        total_time = raw_time+raw_unary_time+end_posprocess_time+end_parenthesized_time+end_merge_retags_time
        #We do no need to count the time of this script as adapting the NCRFpp for this would be straight forward
        total_time-= time_mt2st
        
        print ("Total time:",round(total_time,2)) 
        print ("Sents/s:   ", round(len(gold_trees)/(total_time),2))
        
        #We are also saving the output in a sequence tagging format, in case we want to perform
        #a further analysis

        with codecs.open(args.output+".seq_lu","w") as f_out_seq_lu:
            for (sentence, sentence_preds) in zip(sentences,preds):
                for ((w, pos), l) in zip(sentence, sentence_preds):
                    f_out_seq_lu.write("\t".join([w,pos,l])+"\n")
                f_out_seq_lu.write("\n")
    
        
        #Computing additional metrics: accuracy
        if args.retagger:
            enriched_preds = get_enriched_labels_for_retagger(preds, new_unary_preds)
            flat_preds = flat_list(enriched_preds)
        else:
            flat_preds = flat_list(preds)
      #NEEDs to be adapted with the multitask setup
      #  print "Accuracy:  ",  round(accuracy_score(gold_labels, flat_preds),4)

        
