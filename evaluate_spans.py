from argparse import ArgumentParser
from nltk.tree import Tree
from collections import Counter
import codecs
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import os


def get_tree_spans(tree, root, ignore_non_terminal=False):
    
    spans = []
    if type(tree) != type(""):
        if not root:  
               
            if len(tree.leaves()) == 1:
                if "@" in tree.label():
                    spans = [(tree.leaves(), tree.label())]
            else:  
                if ignore_non_terminal:
                    symbol = "-"
                else:
                    symbol = tree.label()
                spans = [(tree.leaves(), symbol)]
            
        for child in tree:            
            if type(child) != type(""):
                spans.extend(get_tree_spans(child, False))

    return spans


def performance_on_non_terminals(pred_trees, gold_trees):

    span_performance_by_len = {}
    
    non_terminals = {}
    
    for (pred_tree, gold_tree) in zip(predicted_trees, gold_trees):

        predicted_tree = Tree.fromstring(pred_tree,remove_empty_top_bracketing=True)
        gold_tree = Tree.fromstring(gold_tree,remove_empty_top_bracketing=True)
        predicted_tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="@")    
        gold_tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="@")  
        gold_spans = get_tree_spans(gold_tree,True)
        predicted_spans = get_tree_spans(predicted_tree,True)        
        remaining_predicted_spans = copy.deepcopy(predicted_spans)

        for gold_span in gold_spans: #A list of tuple (list of tokens, non-terminal)     
            
            uppermost_non_terminal = gold_span[1].split("@")[0]              
            if uppermost_non_terminal not in non_terminals:
                non_terminals[uppermost_non_terminal] = {"tp":0., "fn":0., "fp":0.}    

            if gold_span in remaining_predicted_spans: 
                non_terminals[uppermost_non_terminal]["tp"]+=1           
                try:
                    remaining_predicted_spans.remove(gold_span)
                except ValueError:
                    pass
            else:
                non_terminals[uppermost_non_terminal]["fn"]+=1
             
        #These are false positives 
        for missing in remaining_predicted_spans:  
            uppermost_missing_span = missing[1].split("@")[0]
            
            if uppermost_missing_span not in non_terminals:
                non_terminals[uppermost_missing_span] = {"tp":0., "fn":0., "fp":0.}    

            non_terminals[uppermost_missing_span]["fp"]+=1
    
    scores = {}
    #Computing precision and recall
    for nt in non_terminals:
        tp = non_terminals[nt]["tp"]
        fn = non_terminals[nt]["fn"]
        fp = non_terminals[nt]["fp"]
        scores[nt] = {"p": 0 if (tp + fp) == 0 else  tp / (tp + fp) ,
                               "r": 0 if (tp+fn) == 0 else tp / (tp+fn)}
 
    return scores


def performance_on_span_len(pred_trees, gold_trees):
     
    spans_by_len = {}
    
    for (pred_tree, gold_tree) in zip(predicted_trees, gold_trees):

        predicted_tree = Tree.fromstring(pred_tree,remove_empty_top_bracketing=True)
        gold_tree =Tree.fromstring(gold_tree,remove_empty_top_bracketing=True)
        predicted_tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="@")    
        gold_tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="@")  
        gold_spans = get_tree_spans(gold_tree,True)
        predicted_spans = get_tree_spans(predicted_tree,True)      
        remaining_predicted_spans = copy.deepcopy(predicted_spans)
       
        for gold_span in gold_spans: #A list of tuple (list of tokens, non-terminal)      
            gold_span_len =  len(gold_span[0])       
            if gold_span_len not in spans_by_len:
                spans_by_len[gold_span_len] = {"tp":0., "fn":0., "fp": 0.}

            if gold_span in remaining_predicted_spans: 
                spans_by_len[gold_span_len]["tp"]+=1
                
                try:
                    remaining_predicted_spans.remove(gold_span)
                except ValueError:
                    pass
            else:
                spans_by_len[gold_span_len]["fn"]+=1
        
        #These are false positives        
        for missing in remaining_predicted_spans:  
            span_len = len(missing[0])
            if span_len not in spans_by_len:
                spans_by_len[span_len] = {"tp":0., "fn":0., "fp": 0.}
            spans_by_len[span_len]["fp"]+=1
           
    scores = {}
    #Computing precision and recall
    for span_length in spans_by_len:
        tp = spans_by_len[span_length]["tp"]
        fn = spans_by_len[span_length]["fn"]
        fp = spans_by_len[span_length]["fp"]
        scores[span_length] = {"p": 0 if (tp + fp) == 0 else  tp / (tp + fp) ,
                               "r": 0 if (tp+fn) == 0 else tp / (tp+fn)}
    
    return scores #span_scores_by_len



def avg_non_terminal_len(path_file):
    
    with codecs.open(path_file) as f:    
        gold_trees = f.readlines()
        non_terminals = []
        nt_lengths = {}
    
    for gold_tree in gold_trees:
        gold_tree =Tree.fromstring(gold_tree,remove_empty_top_bracketing=True)
        gold_tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="@")  
        gold_spans = get_tree_spans(gold_tree,True)
        
        for span_text, span_nt in gold_spans:
            
            uppermost_span_nt = span_nt.split("@")[0]
            non_terminals.append(uppermost_span_nt)
            if uppermost_span_nt not in nt_lengths:
                nt_lengths[uppermost_span_nt] = []
            
            nt_lengths[uppermost_span_nt].append(len(span_text))

    for key in nt_lengths:
        nt_lengths[key] = float(sum(nt_lengths[key]) / len(nt_lengths[key]))
    nt_counter = Counter(non_terminals)

    return nt_counter, nt_lengths

def model2legends(model_name):
    
    if model_name.lower().startswith("elmo"):
        return "ELMo"
    elif model_name.lower().startswith("bert"):
        return "BERT"
    elif model_name.lower().startswith("random"):
        return "Random"
    elif model_name.lower().startswith("glove"):
        return "GloVe"
    elif model_name.lower().startswith("wiki-news"):
        return "FastText"
    elif model_name.lower().startswith("google"):
        return "Word2vec"
    elif model_name.lower().startswith("dyer"):
        return "S. word2vec"
    else:
        return model_name


if __name__ == '__main__':
    
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--predicted",
                            help="Path to the directory containing the predicted input files parenthesized format", 
                            default=None)
    arg_parser.add_argument("--gold",
                            help="Path to the gold input file in parenthesized format")
    arg_parser.add_argument("--span_length_threshold", default=25)
    args = arg_parser.parse_args()
    
    
    span_lengths = []
    ps = []
    rs = []
    f1s = []
    models = []
    sizes = []
    
    #Non terminals and variables to compute the plot f1-score/non-terminal
    nts = []
    nts_occ = []
    ps_nt = []
    rs_nt = []
    f1s_nt = []
    models_nt = []
    sizes_nt = []
    
    ##############################################################################
    # Computing the average lengths for the gold spans                           #
    # (if unary chains are presetns, we only consider the uppermost element)     #
    ##############################################################################
    
    with codecs.open(args.gold) as f_gold:    
        gold_trees = f_gold.readlines()
    nt_counter, nt_lengths = avg_non_terminal_len(args.gold)

    ###############################################################################
    # Reading the predicted parenthesized files to do the plots
    ###############################################################################

    input_files = sorted([args.predicted+os.sep+f 
                   for f in os.listdir(args.predicted)])
    
    for input_file in input_files:
        print ("Processing file", input_file)
        with codecs.open(input_file) as f_input:
            
            predicted_trees = f_input.readlines()
            model_name = input_file.rsplit("/",1)[1]
            model_name = model_name.replace(".test.outputs.txt","")        
            span_performance_by_len = performance_on_span_len(predicted_trees, gold_trees)
            nt_performance = performance_on_non_terminals(predicted_trees, 
                                                                 gold_trees)
           # f = lambda x: 0 if x[1] == 0 else round(x[0] / x[1],4)
            f1_score = lambda p,r : 0 if p== 0 and r == 0 else round(2*(p*r) / (p+r),4)
            
            for span_len in sorted(span_performance_by_len):
                if span_len > args.span_length_threshold: continue
                p = span_performance_by_len[span_len]["p"] 
                r = span_performance_by_len[span_len]["r"]
                f1 =  f1_score(p,r) 
                span_lengths.append(span_len)
                ps.append(p)
                rs.append(r)
                f1s.append(f1)
                models.append(model2legends(model_name))
                sizes.append(14)

            for non_terminal, occ in nt_counter.most_common(7):
                p = nt_performance[non_terminal]["p"]
                r = nt_performance[non_terminal]["r"]
                f1 =  f1_score(p,r)
                nts.append(non_terminal+"("+str(round(nt_lengths[non_terminal],1))+")")
                nts_occ.append(occ)
                ps_nt.append(p)
                rs_nt.append(r)
                f1s_nt.append(f1)
                models_nt.append(model2legends(model_name))

    
    d = {"span_length": span_lengths,
         "precision": ps,
         "recall": rs,
         "f1-score": f1s,
         "model": models}
    
    data = pd.DataFrame(d)
        
    d_nt = {"non_terminal": nts,
            "non_terminal_occ": nts_occ,
            "precision": ps_nt,
            "recall": rs_nt,
            "f1-score": f1s_nt,
            "model": models_nt }

    data_nt = pd.DataFrame(d_nt)

    data_nt.sort_values(by=['model',"non_terminal_occ"],  inplace=True, ascending=[0,0])

#########################################################################
#                      PLOT F1-SCORE/SPAN_LENGTH                        #
#########################################################################

markers = ['o','.',',','*','v','D','h','X','d','^','o','o','<','>']
sns.set(style="ticks", rc={"lines.linewidth": 5, 'lines.markersize': 14})
palette = dict(zip(sorted(set(models)), sns.color_palette()))
ax = sns.lineplot(x="span_length", y="f1-score",
             hue="model", style="model",
             data=data,
             markers=True,
             palette=palette,
             dashes=False)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:],loc='upper center', ncol= int(len(set(models)) / 2),
          bbox_to_anchor=(0.5, 1.25))
plt.setp(ax.get_legend().get_texts(), fontsize='35') 
ax.tick_params(labelsize=30)
ax.set_xlabel("Span length",fontsize=35)
ax.set_ylabel("F1-score",fontsize=35)
plt.show()

#########################################################################
#           PLOT F1-SCORE/NON-TERMINAL SYMBOL (avg length)              #
#########################################################################

ax = sns.barplot(x="non_terminal", y="f1-score",
             hue="model",
             data=data_nt,
             palette=palette)

handles, labels = ax.get_legend_handles_labels()
labels = ["("+str(idl)+") "+l for idl,l in enumerate(labels,1)] 
ax.legend(handles=handles, labels=labels,loc='upper center', ncol= int(len(set(models)) / 2),
          bbox_to_anchor=(0.5, 1.25))
plt.setp(ax.get_legend().get_texts(), fontsize='35') # for legend text
ax.tick_params(labelsize=30,rotation=15)
ax.set_xlabel("Span head non terminal (avg span length)",fontsize=35)
ax.set_ylabel("F1-score",fontsize=35)

plt.show()
