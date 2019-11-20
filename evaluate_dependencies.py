from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter, OrderedDict
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import copy
import os
from _curses import tparm



class WordConll(object):
    
    def __init__(self, idx, word, head, deprel):
        self.idx = idx
        self.word = word
        self.head = head
        self.deprel = deprel


def dependency_displacements(tree, unlabeled=False):
    
    distances = []
    for word in tree[1:]:
        distance = word.head - word.idx
        if unlabeled:
            distances.append(distance)
        else:
            distances.append(str(distance)+"_"+word.deprel)
    return distances
    

def word_relations(tree, unlabeled=True):
    
    relations = []
    for word in tree[1:]:
        if unlabeled:
            relations.append(word.deprel)
        else:
            relations.append(str(word.head)+"_"+word.deprel)
    
    return relations
       


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


def read_conllu(conllu_sentence):
    

        lines = conllu_sentence.split("\n")
        tree = [WordConll(0,"-ROOT-","_","_")]
        for l in lines:
            if l.startswith("#"):continue
            if l != "":
                ls = l.split("\t")
                if "." in ls[0] or "-" in ls[0]: continue
                word_conllu= WordConll(int(ls[0]), ls[1], int(ls[6]), ls[7])
                tree.append(word_conllu)
        
        return tree

        

def elements_to_plot_from_conll(path_file, unlabeled):    
    
    with open(path_file) as f:
        sentences = f.read().split("\n\n")
        distances = []
        relations = []
        trees = []
        for s in sentences[:-1]:
            tree = read_conllu(s) 
            trees.append(tree)
            distances.extend(dependency_displacements(tree, unlabeled))
            relations.extend(word_relations(tree, unlabeled))    
    return distances, relations


def dependency_head_performance(gold_labels, pred_labels):
 
    relations = {}
 
    assert(len(gold_labels),len(pred_labels))
    for gold_element, pred_element in zip(gold_labels, pred_labels):
     
        gold_head = int(gold_element.split("_")[0]) 
        gold_relation =  gold_element.split("_")[1] 
        pred_head = int(pred_element.split("_")[0]) 
        pred_relation = pred_element.split("_")[1]
 
        if gold_relation not in relations:       
            relations[gold_relation] = {"total_desired":0.,"total_predicted":0.,"correct":0.}
 
        if pred_relation not in relations:
            relations[pred_relation] = {"total_desired":0.,"total_predicted":0.,"correct":0.}
         
        relations[gold_relation]["total_desired"]+=1
        relations[pred_relation]["total_predicted"]+=1
         
        if gold_element == pred_element:
            relations[gold_relation]["correct"]+=1
         
    scores ={}
    for rel in relations:
        desired = relations[rel]["total_desired"];
        predicted = relations[rel]["total_predicted"]
        correct = relations[rel]["correct"]
        p = 0 if predicted == 0 else correct / predicted
        r = 0 if desired == 0 else correct / desired
        scores[rel] = {"p": p, "r": r}

    return scores 


def displacement_labeled_performance(gold_distances, pred_distances):
    
    
    distances = {}
    assert(len(gold_distances),len(pred_distances))
    for gold_element, pred_element in zip(gold_distances, pred_distances):
    
        gold_distance = int(gold_element.split("_")[0]) 
        pred_distance = int(pred_element.split("_")[0]) 
        
        if gold_distance not in distances:       
            distances[gold_distance] = {"total_desired":0.,"total_predicted":0.,"correct":0.}
 
        if pred_distance not in distances:
            distances[pred_distance] = {"total_desired":0.,"total_predicted":0.,"correct":0.}
         
        distances[gold_distance]["total_desired"]+=1
        distances[pred_distance]["total_predicted"]+=1
         
        if gold_element == pred_element:
            distances[gold_distance]["correct"]+=1

    #Computing precision and recall

    scores ={}
    for d in distances:
        desired = distances[d]["total_desired"];
        predicted = distances[d]["total_predicted"]
        correct = distances[d]["correct"]
        p = 0 if predicted == 0 else correct / predicted
        r = 0 if desired == 0 else correct / desired
        scores[d] = {"p": p, "r": r}
 
    return scores 
    



if __name__ == '__main__':
    
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--predicted", 
                            help="Path to the directory containing the predicted input files in conllu format", 
                            default=None)
    arg_parser.add_argument("--gold",
                            help="Path to the gold file in conll format")
    arg_parser.add_argument("--unlabeled", 
                            default=False,
                            action="store_true",
                            help="Ignores dependency types")
    
    args = arg_parser.parse_args()
    
    ############################################################################
    # Computing:
    # 1. Dependency displacements for the gold trees (gold_distances)
    # 2. Gold dependency relations (no head index taken into account)
    ############################################################################    
    all_distances = []
    idx_distances = OrderedDict()

    gold_distances, gold_relations = elements_to_plot_from_conll(args.gold, args.unlabeled)         
    gold_relations_counter = Counter(gold_relations)

    input_files = sorted([args.predicted+os.sep+f 
                   for f in os.listdir(args.predicted)])    

    #Variables to compute the dependency displacement scores
    distances = []
    distances_precision  = []
    distances_recall = []
    distances_f1 = []
    distances_models = []
    
    #Variables to compute the performance on the most common dependency relations
    relations = []
    relations_precision = []
    relations_recall = []
    relations_f1 = []
    relations_models = []
    relations_occ = []

    for input_file in input_files:        
    
        pred_distances, pred_relations = elements_to_plot_from_conll(input_file, args.unlabeled)
        model_name = input_file.rsplit("/",1)[1]
        model_name = model_name.replace(".test.outputs.txt","")   
        print ("Processing file:", input_file)
        ########################################################################
        # Performance on different dependency relations 
        # (not penalizing the index of the head)
        ########################################################################
        rel2idx = {e:idx for idx,e in enumerate(sorted(set(gold_relations).union(set(pred_relations))))}
        idx2rel = {rel2idx[e]:e for e in rel2idx}
        
        if args.unlabeled:
            indexed_gold_relations = [rel2idx[g] for g in gold_relations]
            indexed_pred_relations = [rel2idx[p] for p in pred_relations]
            precision, recall, f1, support = precision_recall_fscore_support(indexed_gold_relations, indexed_pred_relations)
            
            for relation, occurrences in gold_relations_counter.most_common(n=7):
                
                idx_relation = rel2idx[relation]
                relations.append(relation)
                relations_precision.append(precision[idx_relation])
                relations_recall.append(recall[idx_relation])
                relations_f1.append(f1[idx_relation])
                relations_models.append(model2legends(model_name))

        else:

            relations_performance = dependency_head_performance(gold_relations, pred_relations)             
            _, aux_gold_relations = elements_to_plot_from_conll(args.gold, True)         
            aux_gold_relations_counter = Counter(aux_gold_relations)
            f1_score = lambda p,r : 0 if p== 0 and r == 0 else round(2*(p*r) / (p+r),4)
            
            for relation, occurrences in aux_gold_relations_counter.most_common(n=7): 
                relations.append(relation)
                relations_occ.append(occurrences)
                relation_p = relations_performance[relation]["p"]
                relation_r = relations_performance[relation]["r"]
                relations_precision.append(relation_p)
                relations_recall.append(relation_r)
                relations_f1.append(f1_score(relation_p, relation_r))
                relations_models.append(model2legends(model_name))
    
        #########################################################################
        # Performance for each displacement, labeled or unlabeled
        #########################################################################
        if args.unlabeled:        
            labelsi = {e:idx for idx,e in enumerate(sorted(set(gold_distances).union(set(pred_distances))))}
            ilabels = {labelsi[e]:e for e in labelsi}
            aux_gold_distances = [labelsi[g] for g in gold_distances]
            aux_pred_distances = [labelsi[p] for p in pred_distances]
            precision, recall, f1, support = precision_recall_fscore_support(aux_gold_distances, aux_pred_distances)
            
        else:
            
            distances_performance = displacement_labeled_performance(gold_distances, pred_distances)
            precision =[]
            recall = []
            f1 = []
            support = []
            f1_score = lambda p,r : 0 if p== 0 and r == 0 else round(2*(p*r) / (p+r),4)
            
            labelsi = {e:idx for idx,e in enumerate(  sorted(set(list(map(int,distances_performance)))  ))}
            ilabels = {labelsi[e]:e for e in labelsi}
            
            for distance in sorted(list(map(int,distances_performance))):    
                distance_p = distances_performance[distance]["p"]
                distance_r = distances_performance[distance]["r"]
                precision.append(distance_p)
                recall.append(distance_r)
                f1.append(f1_score(distance_p, distance_r))
                support.append(None)
            
            
        for idxe, (p,r,f,s) in enumerate(zip(precision,recall,f1,support)):

            if abs(int(ilabels[idxe])) <= 20:
                distances.append(ilabels[idxe])
                distances_precision.append(p)
                distances_recall.append(r)
                distances_f1.append(f)
                distances_models.append(model2legends(model_name))
        
    d = {"distances": distances,
             "precision": distances_precision,
             "recall": distances_recall,
             "f1-score": distances_f1,
             "model": distances_models}
        
    data = pd.DataFrame(d)    
        
    
    ############################################################################
    #                 PLOTTING DEPENDENCY DISPLACEMENTS                        #
    ############################################################################
            
    markers = ['o','.',',','*','v','D','h','X','d','^','o','o','<','>']
    sns.set(style="ticks", rc={"lines.linewidth": 5, 'lines.markersize': 14})
    palette = dict(zip(sorted(set(distances_models)), sns.color_palette()))
    
    ax = sns.lineplot(x="distances", y="f1-score",
                 hue="model", style="model",
                 data=data,
                 palette=palette,
                 markers=True,
                 dashes=False)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:],loc='upper center', ncol= int(len(set(distances_models)) / 2),
              bbox_to_anchor=(0.5, 1.25))
    plt.setp(ax.get_legend().get_texts(), fontsize='35') # legend text size
    
    ax.tick_params(labelsize=30)
    ax.set_xlabel("Dependency displacement",fontsize=35)
    ax.set_ylabel("F1-score",fontsize=35)
    
    plt.show()
    
    ############################################################################
    #                PLOTTING F1-SCORE/DEPENDENCY-RELATIONS                    #
    ############################################################################
    
    d_rel = {"relations": relations,
             "relations_occ": relations_occ,
             "precision": relations_precision,
             "recall": relations_recall,
             "f1-score": relations_f1,
             "model": relations_models}
    
    data_rel = pd.DataFrame(d_rel)    
    
    data_rel.sort_values(by=['model','relations_occ'],  inplace=True, ascending=[0,0])
    
    ax = sns.barplot(x="relations", y="f1-score",
                 hue="model", 
                 data=data_rel,
                 palette=palette)

    handles, labels = ax.get_legend_handles_labels()
    labels = ["("+str(idl)+") "+l for idl,l in enumerate(labels,1)] 
    ax.legend(handles=handles, labels=labels,loc='upper center', ncol= int(len(set(distances_models)) / 2),
              bbox_to_anchor=(0.5, 1.25))
    plt.setp(ax.get_legend().get_texts(), fontsize='35') # for legend text
    
    ax.tick_params(labelsize=30,rotation=15)
    ax.set_xlabel("Dependency relation",fontsize=34)
    ax.set_ylabel("F1-score",fontsize=35)    
    
    plt.show()

    
    