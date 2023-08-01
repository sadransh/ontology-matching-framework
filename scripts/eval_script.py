import numpy as np
import pandas as pd
import pickle
import os
import json

from tqdm import tqdm
from sentence_transformers.util import cos_sim
import argparse
import ast
import re

from deeponto.onto.mapping import OntoMappings
from collections import defaultdict
import collections.abc
from deeponto import SavedObj
from deeponto.evaluation.align_eval import local_rank_eval
from deeponto.evaluation.eval_metrics import *


def get_max_sim(srcent,tgent,layer_name='encoder'):
    """calculate max similarity between two embeddings

    Args:
        srcent (dict): source entity
        tgent (dict): target entity
        layer_name (str, optional): layer name of . Defaults to 'encoder'.

    Returns:
        _type_: max similarity score
    """
    list_src_ents=[srcent[layer_name]] + [i[layer_name] for i in srcent['Syn']]+ [i[layer_name] for i in srcent['Def']]
    list_tg_ents=[tgent[layer_name]]+ [i[layer_name] for i in tgent['Syn']]+ [i[layer_name] for i in tgent['Def']]
    list_src_fsn = [srcent['fsn'].lower()]+[i['fsn'].lower() for i in srcent['Syn']] + [i['fsn'].lower() for i in srcent['Def']]
    list_tg_fsn = [tgent['fsn'].lower()]+[i['fsn'].lower() for i in tgent['Syn']] + [i['fsn'].lower() for i in tgent['Def']]
    if any(x in list_tg_fsn for x in list_src_fsn):
        return 1.0
    return max([cos_sim(i,j).item() for i in list_src_ents for j in list_tg_ents])

def get_embedding_score(src_snomed,tgt_ont,ln,targetsep,src_ontology, tgt_ontology):
    """prepare embeddings for evaluation

    Args:
        src_snomed (_type_): source entity
        tgt_ont (_type_): target entity
        ln (_type_): layer name
        targetsep (_type_): separator for identifyer and code   
        src_ontology (_type_): source ontology
        tgt_ontology (_type_): target ontology

    Returns:
        _type_: a dictionary of similarity scores
    """
    all_source_sibs = list(set([src_snomed.split(":")[-1]])) 

    all_target_sibs = list(set([tgt_ont.split(":")[-1]]))

    tmp_dict = {}
    for i in all_source_sibs:
        tmp_dict["snomed:"+i] = tmp_dict.get("snomed:"+i,{})
        for j in all_target_sibs:

            sim_score = -1 if None in [src_ontology.get(i,None),tgt_ontology.get(j,None)] else get_max_sim(src_ontology[i],tgt_ontology[j],layer_name=ln)
            tmp_dict["snomed:"+i][targetsep+j] = \
            max(tmp_dict["snomed:"+i].get(targetsep+j,0.0000),sim_score)

    return tmp_dict



def flatten(seq):
    """ flatten a list"""
    return [val for sublist in seq for val in sublist]

def extract_text_for_extension(fully_specified_name):
    """ extract text from a fully_specified_name"""
    return " ".join([re.sub(r"\([^)]*\)", "", fully_specified_name).rstrip()]+re.findall(r'\([^)]*\)', fully_specified_name)[:-1]).rstrip()


def extract_text(fully_specified_name):
    """
    This function extracts the SMOMED name given a SNOMED CT definitions.
    Basically, as all the definitions have their type infront of them, we can extract the name by removing the type.
    (i.e. Disease caused by severe acute respiratory syndrome coronavirus 2 (disorder) --> Disease caused by severe acute respiratory syndrome coronavirus 2))
    Args:
        fully_specified_name (String): SNOMED CT definitions.
    Returns:
        String: SMOMED name.
    """
    valid_types = ['state of matter', 'release characteristic', 'link assertion', 'biological function', 'inactive concept', 'owl metadata concept', 'transformation', 'supplier', 'product name','administration method', 'small pieces', 'category', 'symptom', 'weakness', 'organ', 'role', 'operation', 'disorder', 'procedure', 'finding', 'organism', 'body structure', 'substance', 'product', 'physical object', 'qualifier value', 'situation', 'observable entity', 'event', 'medicinal product', 'medicinal product form', 'occupation', 'morphologic abnormality', 'clinical drug', 'regime/therapy', 'specimen', 'assessment scale', 'environment', 'attribute', 'cell', 'person', 'navigational concept', 'geographic location', 'record artifact', 'cell structure', 'disposition', 'ethnic group', 'foundation metadata concept', 'tumor staging', 'namespace concept', 'religion/philosophy', 'dose form', 'physical force', 'combined site', 'context-dependent category', 'role', 'isbt symbol', 'property', 'core metadata concept', 'unit of presentation', 'staging scale', 'contextual qualifier', 'surface region', 'basic dose form', 'general', 'special concept', 'social concept', 'life style', 'intended site', 'clinical', 'racial group', 'machine', 'administrative concept']
    added_types = ['contextual qualifier' , 'motion picture' , 'mine' , 'general']
    names = fully_specified_name.split("(")
    fsn = []
    for name in names:
        stripped_name = name.strip()
        try:
            if stripped_name[-1]==")" and stripped_name[:-1] in valid_types +  added_types :
                pass
            else:
                fsn.append(name)
        except:
            continue
    fsn = "(".join(fsn)
    return fsn.strip()


def clean_text_func(text):
    """ clean text by removing special characters"""
    return text.lower().replace(';', '').replace('.', ' ').replace('/', ' ').\
            replace('-', ' ').replace('&', ' ').replace(")", "").replace(",", " ").\
            replace("(", "").replace("'", ' ').replace("[", "").replace("]","").replace('  ', ' ').strip()

def load_resource(task):
    """ load resources for evaluation"""
    if task == 'body': 

        with open(os.path.join("../data/ontology_json",'snomed_body_concepts.json'), 'r') as f:
            source_concepts = json.load(f)
        with open(os.path.join("../data/ontology_json",'fma_concepts_body.json'), 'r') as f:
            tgt_concepts = json.load(f)
        test_data = list(source_concepts.keys())            

        output_iden = 'fma:'
        
        
    if task == 'phar': 

        with open(os.path.join("../data/ontology_json",'snomed_phar_concepts.json'), 'r') as f:
            source_concepts = json.load(f)
        with open(os.path.join("../data/ontology_json",'ncit_phar_concepts.json'), 'r') as f:
            tgt_concepts = json.load(f)
            
        test_data = list(source_concepts.keys())

        output_iden = 'ncit_largebio:'

    if task == 'neop': 

        with open(os.path.join("../data/ontology_json",'snomed_neop_concepts.json'), 'r') as f:
            source_concepts = json.load(f)
        with open(os.path.join("../data/ontology_json",'ncit_neop_concepts.json'), 'r') as f:
            tgt_concepts = json.load(f)

        test_data = list(source_concepts.keys())

        output_iden = 'ncit_largebio:'
    return source_concepts,tgt_concepts,test_data,output_iden

def get_common_PRF1ACC(task,fname,model_output,no_em=False):
    """ get common metrics based on softmax of the last layer"""
    clean_text = True
    checkpoint = fname.split("-")[-1]

    with open(f'{fname}/results-{checkpoint}-{task}-{model_output}-0.json', 'r') as f:
        res = json.load(f)

    pred_c1, pred_prob_c1, fsn_pred_c1 = [res[i]['pred_c1'] for i in range(len(res))], \
    [res[i]['pred_prob_all'] for i in range(len(res))], [res[i]['pred_fsn_all'] for i in range(len(res))]


    source_concepts,tgt_concepts,test_data,output_iden = load_resource(task)

    if task in ['body','phar','neop']:
        with open(f"../data/ontology_json/snomed_ct_concepts20160901.json", "r") as fp:
            old_snomed_concepts = json.load(fp)
        for k,v in source_concepts.items():
            if k in old_snomed_concepts.keys():
                old_values = old_snomed_concepts.get(k,'None')
                if old_values !='None' and extract_text_for_extension(v['FullySpecifiedName']) != extract_text_for_extension(old_values['FullySpecifiedName']) :
                    v['Synonyms'].append(extract_text_for_extension(old_values['FullySpecifiedName']))


    fsn_to_id = {}
    for code,val in tgt_concepts.items():
        descriptions = val.get('Preferred_name',[]) + val['Label'] + val['Synonym']

        
        if clean_text:
            descriptions = list(set([clean_text_func(i.lower()) for i in descriptions if i!='']))

        else: 
            descriptions = list(set([i.lower() for i in descriptions if i!='']))
        for d in descriptions:
            if fsn_to_id.get(d):
                fsn_to_id[d] = fsn_to_id[d] + [code]
            else:
                fsn_to_id[d] = [code]
        
    ### smart ids to true codes 

    with open(os.path.join("../data",'omversion_fma_to_hierarchicalId.json'), 'r') as f:
        fma_tids = json.load(f)
    with open(os.path.join("../data",'omversion_ncit_to_hierarchicalId.json'), 'r') as f:
        ncit_tids = json.load(f)

    dic_comb = {**fma_tids, **ncit_tids}
    sid_to_id = {val:key for key,val in dic_comb.items()}

    if task=='body':
        prompt = '1-'
    else:
        prompt = '2-'

            
    inp_test_all = []


    for i in test_data:
        inp_test_all.append({
            'input':source_concepts.get(i)["FullySpecifiedName"] if source_concepts.get(i) else '',
            'input_snomed_code' : i,
        })


    # %%
    def convert2json(codes, prob):
        res = {}

        for i in range(len(codes)):
            src_code = 'snomed:'+inp_test_all[i]['input_snomed_code']
            if len(codes[i])==1 and codes[i][0]=='' or len(codes[i])==0:
                pass
            else:
                d = {}
                for k in range(len(codes[i])):
                    if codes[i][k]!='':
                        # print(i,codes[i],prob[i])
                        pred_code = codes[i][k]
                        if output_iden+pred_code not in d:
                            d[output_iden+pred_code] = prob[i][k]
                    if len(d)!=0:
                        res[src_code] = d
        return res    

    def drop_dup(pred_fsn_c1_codes,pred_fsn_c1_probs):
        
        all_predicted_codes,all_predicted_codes_prob =[],[]
        
        for raw_codes,raw_scores in zip(pred_fsn_c1_codes,pred_fsn_c1_probs):
        ## sorting and removing duplicates from the raw codes based on scores
            if len(raw_codes)!=0:
                scores = np.array(raw_scores)
                sorted_indices = np.flip(np.argsort(scores))
                sorted_codes_dup = np.array(raw_codes)[sorted_indices].tolist()  
                sorted_scores_dup = np.array(raw_scores)[sorted_indices].tolist()

                ### remove duplicates  
                sorted_codes, sorted_scores = [sorted_codes_dup[0]], [sorted_scores_dup[0]]

                for l,m in zip(sorted_codes_dup[1::], sorted_scores_dup[1::]):
                    if l not in sorted_codes:
                        sorted_codes.append(l)
                        sorted_scores.append(m)
            else:
                sorted_codes = ['']
                sorted_scores = [0]

            # extracting predicted codes
            all_predicted_codes.append(sorted_codes)
            all_predicted_codes_prob.append(sorted_scores)
        return all_predicted_codes,all_predicted_codes_prob

    # flattens data

    pred_fsn = [flatten(i[0]) for i in fsn_pred_c1]
    pred_fsn_prob = [flatten(i[1]) for i in fsn_pred_c1]


    ## logic with no exact match
    out = [([sid_to_id.get(prompt+c[0])],[p[0]]) if sid_to_id.get(prompt+c[0]) else ([''],[0]) for c,p in zip(pred_fsn,pred_fsn_prob)]

    pred_fsn_c1_codes = [i[0] for i in out]
    pred_fsn_c1_probs = [i[1] for i in out]
        
    results_fsn_1beam = convert2json(pred_fsn_c1_codes, pred_fsn_c1_probs)


    ## Exact match logic

    pred_EM = []
    pred_EM_prob = []

    for i,pred,prob in zip(test_data,pred_fsn_c1_codes,pred_fsn_c1_probs):
        out_code = []
        conc = source_concepts.get(i)
        if conc:

            inp_list = list(set([extract_text_for_extension(conc["FullySpecifiedName"])] + conc['Synonyms'] + conc['Definitions']))

            for k in inp_list:
                if clean_text:
                    match = fsn_to_id.get(clean_text_func(k.lower()))
                else:
                    match = fsn_to_id.get(k.lower())
                if match:
                    out_code.extend(match)


        ### remove dup
        out_code = list(set(out_code))
        pred_EM.append(out_code)
        pred_EM_prob.append([1]*len(out_code))                                                               

    ##prepares the data    
    out = [([sid_to_id.get(prompt+c[0])],[p[0]]) if sid_to_id.get(prompt+c[0]) else ([''],[0]) for c,p in zip(pred_fsn,pred_fsn_prob)]

    pred_fsn_c1_codes = [i[0] for i in out]
    pred_fsn_c1_probs = [i[1] for i in out]

    pred_fsn_c1_codes = [i+j for i,j in zip(pred_EM, pred_fsn_c1_codes)]
    pred_fsn_c1_probs = [i+j for i,j in zip(pred_EM_prob, pred_fsn_c1_probs)]

    pred_fsn_c1_codes,pred_fsn_c1_probs =  drop_dup(pred_fsn_c1_codes,pred_fsn_c1_probs)
        
    results_fsn_1beam_em = convert2json(pred_fsn_c1_codes, pred_fsn_c1_probs)


    eval_data = results_fsn_1beam_em if no_em==False else results_fsn_1beam

    source_split = "/"
    if task == "neop":
        splittgt="#"
        tag = "ncit_largebio:"
        file = "snomed2ncit.neoplas"
    elif task == "phar":
        splittgt='#'
        tag = "ncit_largebio:"
        file = "snomed2ncit.pharm"
    elif task == "body":
        tag = "fma:"
        splittgt="/"
        file = "snomed2fma.body"


    ## test data
    data = pd.read_csv(f'../data/source_data/UMLS/equiv_match/refs/{file}/unsupervised/test.tsv',sep='\t')

    test = []
    for i in range(len(data)):
        e1 = "snomed:"+data.iloc[i].SrcEntity.split(source_split)[-1]
        e2 = tag+data.iloc[i].TgtEntity.split(splittgt)[-1]
        test.append(e1+'_'+e2)
    test_set = set(test)

    data = pd.read_csv(f'../data/source_data/UMLS/equiv_match/refs/{file}/unsupervised/val.tsv',sep='\t')

    val = []
    for i in range(len(data)):
        e1 = "snomed:"+data.iloc[i].SrcEntity.split(source_split)[-1]
        e2 = tag+data.iloc[i].TgtEntity.split(splittgt)[-1]
        val.append(e1+'_'+e2)
    val_set = set(val)

    precision,recall,fscore, threshold = [],[],[],[]

    for t in np.arange(0.99,1.0001,.0001):
        t = round(t,5)
        res_list = []
        count = 0
        for i in eval_data.keys():
            cands = list(eval_data[i].keys())
            for j in range(len(cands)):
                prob = eval_data[i][cands[j]]
                if prob>=t:
                    res_list.append(i+'_'+cands[j])
            count += 1

        res_set = set(res_list)
        # print(t," .  ",len(res_set))
        P = len(res_set.intersection(test_set))/ len(res_set.difference(val_set) )

        R = len(res_set.intersection(test_set))/len(test_set)
        F1 = (2*P*R)/(P+R)


        precision.append(P)
        recall.append(R)
        fscore.append(F1) 
        threshold.append(t)
    arg_max = np.argmax(fscore)

    print(task,' threshold', round(threshold[arg_max],4), 'Precision:',round(precision[arg_max],4),'Recall:', round(recall[arg_max],4), 'F measure:',round(fscore[arg_max],4))

    return {
        'threshold': round(threshold[arg_max],4),
        'Precision':round(precision[arg_max],4),
        'Recall':round(recall[arg_max],4),
        'F measure:':round(fscore[arg_max],4),
        # "ACC":correct_count/len(test_set)
        }

def get_emb_PRF1(task,fname,model_output,no_em=False,emb_layers='encoder12mean'):
    clean_text = True
    checkpoint = fname.split("-")[-1]

    with open(f'{fname}/results-{checkpoint}-{task}-{model_output}-0.json', 'r') as f:
        res = json.load(f)


    with open(f'{fname}/embeddings-{checkpoint}-{model_output}-{task}.pkl', 'rb') as f:
        embeddings_task = pickle.load(f)
    print("loading embedding for target ....")

    with open(f'{fname}/embeddings-tgt-{checkpoint}-{model_output}-{task}.pkl', 'rb') as f:
        embeddings_tgt = pickle.load(f)

    pred_c1, pred_prob_c1, fsn_pred_c1 = [res[i]['pred_c1'] for i in range(len(res))], \
    [res[i]['pred_prob_all'] for i in range(len(res))], [res[i]['pred_fsn_all'] for i in range(len(res))]


    source_concepts,tgt_concepts,test_data,output_iden = load_resource(task)
   

    with open(f"../data/ontology_json/snomed_ct_concepts20160901.json", "r") as fp:
        old_snomed_concepts = json.load(fp)
    for k,v in source_concepts.items():
        if k in old_snomed_concepts.keys():
            old_values = old_snomed_concepts.get(k,'None')
            if old_values !='None' and extract_text_for_extension(v['FullySpecifiedName']) != extract_text_for_extension(old_values['FullySpecifiedName']) :
                v['Synonyms'].append(extract_text_for_extension(old_values['FullySpecifiedName']))


    fsn_to_id = {}
    for code,val in tgt_concepts.items():
        descriptions = val.get('Preferred_name',[]) + val['Label'] + val['Synonym']

        
        if clean_text:
            descriptions = list(set([clean_text_func(i.lower()) for i in descriptions if i!='']))

        else: 
            descriptions = list(set([i.lower() for i in descriptions if i!='']))
        for d in descriptions:
            if fsn_to_id.get(d):
                fsn_to_id[d] = fsn_to_id[d] + [code]
            else:
                fsn_to_id[d] = [code]
        
    ### h-ids to true codes 

    with open(os.path.join("../data",'omversion_fma_to_hierarchicalId.json'), 'r') as f:
        fma_tids = json.load(f)
    with open(os.path.join("../data",'omversion_ncit_to_hierarchicalId.json'), 'r') as f:
        ncit_tids = json.load(f)

    dic_comb = {**fma_tids, **ncit_tids}
    sid_to_id = {val:key for key,val in dic_comb.items()}

    if task=='body':
        prompt = '1-'
    else:
        prompt = '2-'
    inp_test_all = []
    
    for i in test_data:
        inp_test_all.append({
            'input':source_concepts.get(i)["FullySpecifiedName"] if source_concepts.get(i) else '',
            'input_snomed_code' : i,
        })


    def convert2json(codes, prob):
        res = {}

        for i in range(len(codes)):
            src_code = 'snomed:'+inp_test_all[i]['input_snomed_code']
            if len(codes[i])==1 and codes[i][0]=='' or len(codes[i])==0:
                pass
            else:
                d = {}
                for k in range(len(codes[i])):
                    if codes[i][k]!='':
                        # print(i,codes[i],prob[i])
                        pred_code = codes[i][k]
                        if output_iden+pred_code not in d:
                            d[output_iden+pred_code] = prob[i][k]
                    if len(d)!=0:
                        res[src_code] = d
        return res    

    def drop_dup(pred_fsn_c1_codes,pred_fsn_c1_probs):
        
        all_predicted_codes,all_predicted_codes_prob =[],[]
        
        for raw_codes,raw_scores in zip(pred_fsn_c1_codes,pred_fsn_c1_probs):
        ## sorting and removing duplicates from the raw codes based on scores
            if len(raw_codes)!=0:
                scores = np.array(raw_scores)
                sorted_indices = np.flip(np.argsort(scores))
                sorted_codes_dup = np.array(raw_codes)[sorted_indices].tolist()  
                sorted_scores_dup = np.array(raw_scores)[sorted_indices].tolist()

                ### remove duplicates  
                sorted_codes, sorted_scores = [sorted_codes_dup[0]], [sorted_scores_dup[0]]

                for l,m in zip(sorted_codes_dup[1::], sorted_scores_dup[1::]):
                    if l not in sorted_codes:
                        sorted_codes.append(l)
                        sorted_scores.append(m)
            else:
                sorted_codes = ['']
                sorted_scores = [0]

            # extracting predicted codes
            all_predicted_codes.append(sorted_codes)
            all_predicted_codes_prob.append(sorted_scores)
        return all_predicted_codes,all_predicted_codes_prob

    import numpy as np
    pred_fsn = [flatten(i[0]) for i in fsn_pred_c1]
    pred_fsn_prob = [flatten(i[1]) for i in fsn_pred_c1]


    ## Exact match predictions
    out = [([sid_to_id.get(prompt+c[0])],[p[0]]) if sid_to_id.get(prompt+c[0]) else ([''],[0]) for c,p in zip(pred_fsn,pred_fsn_prob)]

    pred_fsn_c1_codes = [i[0] for i in out]
    pred_fsn_c1_probs = [i[1] for i in out]


    pred_EM = []
    pred_EM_prob = []

    for i,pred,prob in zip(test_data,pred_fsn_c1_codes,pred_fsn_c1_probs):
        out_code = []
        conc = source_concepts.get(i)
        if conc:
            inp_list = list(set([extract_text(conc["FullySpecifiedName"])] + conc['Synonyms'] + conc['Definitions']))

            for k in inp_list:
                if clean_text:
                    match = fsn_to_id.get(clean_text_func(k.lower()))
                else:
                    match = fsn_to_id.get(k.lower())
                if match:
                    out_code.extend(match)


        ### remove dup
        out_code = list(set(out_code))
        pred_EM.append(out_code)
        pred_EM_prob.append([1]*len(out_code))                                                               




    pred_fsn_c1_codes = [i+j for i,j in zip(pred_EM, pred_fsn_c1_codes)]
    pred_fsn_c1_probs = [i+j for i,j in zip(pred_EM_prob, pred_fsn_c1_probs)]

    pred_fsn_c1_codes,pred_fsn_c1_probs =  drop_dup(pred_fsn_c1_codes,pred_fsn_c1_probs)
        
    results_fsn_1beam_em = convert2json(pred_fsn_c1_codes, pred_fsn_c1_probs)


    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    res = {}
    for emb_layer in emb_layers:

        results_fsn_1beam_em_pc = {}

        count_non_exist=0
        for i in results_fsn_1beam_em.keys():
            cands = list(results_fsn_1beam_em[i].keys())
            for j in range(len(cands)):
                # print(i,cands[j],embeddings_task.get(i,None),embeddings_tgt.get(cands[j],None))
                src_node, tgt_node = embeddings_task.get(i.split(":")[-1],None), embeddings_tgt.get(cands[j].split(":")[-1],None)
                deep_update(results_fsn_1beam_em_pc,get_embedding_score(i,cands[j],emb_layer,output_iden,embeddings_task,embeddings_tgt))

        eval_data = results_fsn_1beam_em_pc if no_em==False else results_fsn_1beam_em_pc

        import numpy as np
        import pandas as pd


        source_split = "/"
        if task == "neop":
            splittgt="#"
            tag = "ncit_largebio:"
            file = "snomed2ncit.neoplas"
        elif task == "phar":
            splittgt='#'
            tag = "ncit_largebio:"
            file = "snomed2ncit.pharm"
        elif task == "body":
            tag = "fma:"
            splittgt="/"
            file = "snomed2fma.body"


        ## test data
        data = pd.read_csv(f'../data/source_data/UMLS/equiv_match/refs/{file}/unsupervised/test.tsv',sep='\t')

        test = []
        for i in range(len(data)):
            e1 = "snomed:"+data.iloc[i].SrcEntity.split(source_split)[-1]
            e2 = tag+data.iloc[i].TgtEntity.split(splittgt)[-1]
            test.append(e1+'_'+e2)
        test_set = set(test)

        ## validation data
        data = pd.read_csv(f'../data/source_data/UMLS/equiv_match/refs/{file}/unsupervised/val.tsv',sep='\t')

        val = []
        for i in range(len(data)):
            e1 = "snomed:"+data.iloc[i].SrcEntity.split(source_split)[-1]
            e2 = tag+data.iloc[i].TgtEntity.split(splittgt)[-1]
            val.append(e1+'_'+e2)
        val_set = set(val)

        precision,recall,fscore, threshold = [],[],[],[]

        for t in np.arange(0.9,1.001,.001):
            res_list = []
            count = 0
            for i in eval_data.keys():
                cands = list(eval_data[i].keys())
                for j in range(len(cands)):
                    prob = eval_data[i][cands[j]]
                    if prob>=t:
                        res_list.append(i+'_'+cands[j])
                count += 1

            res_set = set(res_list)

            P = len(res_set.intersection(test_set))/ len(res_set.difference(val_set) )

            R = len(res_set.intersection(test_set))/len(test_set)
            F1 = (2*P*R)/(P+R)
            precision.append(P)
            recall.append(R)
            fscore.append(F1) 
            threshold.append(t)

        arg_max = np.argmax(fscore)


        print(task,emb_layer,'threshold', round(threshold[arg_max],3), 'Precision:',round(precision[arg_max],3),'Recall:', round(recall[arg_max],3), 'F measure:',round(fscore[arg_max],3))


        res[emb_layer] = {
        'emb_layer': emb_layer,
        'threshold': round(threshold[arg_max],4),
        'Precision':round(precision[arg_max],4),
        'Recall':round(recall[arg_max],4),
        'F measure:':round(fscore[arg_max],4),
        }
    return res


def get_embeddings_score(tasks,folder_name,emb_layers):
    """get embedding scores for a given task (similarty between embeddings))"""

    metrics={}



    # body
    src_name = folder_name 
    tgt_name =  folder_name

    model_output = 'ids'


    checkpoint = folder_name.split("-checkpoint-")[-1].replace("/","")
    if not os.path.exists(f'../results/{checkpoint}'):
        os.makedirs(f'../results/{checkpoint}')


    if not os.path.exists(f'../results/{checkpoint}/om'):
        os.makedirs(f'../results/{checkpoint}/om')

    layers= emb_layers
    for task in [tasks]:
        metrics[task]={}

        print("starting task:",task)
        print("loading embedding for task ....")
        with open(f'{src_name}/embeddings-{src_name.split("-")[-1]}-{model_output}-{task}.pkl', 'rb') as f:
            embeddings_task = pickle.load(f)
        print("loading embedding for target ....")
        with open(f'{tgt_name}/embeddings-tgt-{src_name.split("-")[-1]}-{model_output}-{task}.pkl', 'rb') as f:
            embeddings_tgt = pickle.load(f)
        #temp logic to fix 0/1 max/min 


        # pd.read_csv('test-body.cands.tsv', sep='\t')

        src_splitter_elem = "/"
        if task == "neop":
            splitter_elem="#"
            tag = "ncit_largebio:"
            file = "snomed2ncit.neoplas"

        elif task == "phar":
            splitter_elem='#'
            tag = "ncit_largebio:"
            file = "snomed2ncit.pharm"
        elif task == "body":
            tag = "fma:"
            splitter_elem="/"
            file = "snomed2fma.body"

        print("loading candidates ....")
        generic = lambda x: ast.literal_eval(x)
        conv={'TgtCandidates':generic}

        test_set_path = f"../data/source_data/UMLS/equiv_match/refs/{file}/unsupervised/test.cands.tsv"
        df_test = pd.read_csv(test_set_path, converters=conv ,sep='\t', header=0)
        

        for layer in layers:
            
            print("starting layer:",layer)
            preds = defaultdict()


            #preparing data in format consistent with ONtoMappings
            for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
                src_ent = row['SrcEntity']
                preds[src_ent] = preds.get(src_ent,{})
                src = embeddings_task[row['SrcEntity'].split(src_splitter_elem)[-1].split("_")[0]]

                tggt = embeddings_tgt[row['TgtEntity'].split(splitter_elem)[-1]]
                preds[src_ent][row['TgtEntity']]=get_max_sim(src,tggt,layer_name=layer)#print(get_max_sim(src,tggt,layer_name=layer))
            
                for i in row['TgtCandidates']:
                    tgt_candid = embeddings_tgt[i.split(splitter_elem)[-1]]
                    preds[src_ent][i]=get_max_sim(src,tgt_candid,layer_name=layer)

            #creates a dummy object to add predictions
            onto_maps = OntoMappings('src2tgt',0,'')

            onto_maps.map_dict = dict(preds)
            onto_maps.save_instance(f"../results/{checkpoint}/src2tgt_maps-{layer}-{task}")
            print("SAVED: ",f"../results/{checkpoint}/src2tgt_maps-{layer}-{task}")

            results = local_rank_eval(f"../results/{checkpoint}/src2tgt_maps-{layer}-{task}", test_set_path, *[1, 5, 10, 30, 100])
            SavedObj.save_json(results, f"../results/{checkpoint}/om" + f"/{task}-{layer}.results.json")
            metrics[task][layer]=results
    json_object = json.dumps(metrics)

    # Writing to sample.json
    with open(f'../results/{checkpoint}/results-{checkpoint}.json', 'w', encoding='utf-8') as outfile:
        outfile.write(json_object)
    return json_object


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="evaluate the model")
    
    # Add arguments
    parser.add_argument("--task_name", type=str, default="", help="name of the task",required=True)
    parser.add_argument("--score_based",  help="generate metrics for score based method purely on generation of the model",action='store_true')
    parser.add_argument("--embedding_based", help="generate metrics for embedding based approach",action='store_true')
    parser.add_argument("--get_hit_mrr", help="generate hit and mrr",action='store_true')

    parser.add_argument("--prediction_path", type=str, default="", help="path to generated predictions",required=True)

    # Parse the arguments
    args = parser.parse_args()
    
    args.prediction_path = args.prediction_path[:-1] if args.prediction_path[-1]=='/' else args.prediction_path

    all_tasks = ['phar','neop','body'] if 'all' in args.task_name.lower() else [args.task_name.lower()]

    if any([args.score_based,args.embedding_based,args.get_hit_mrr]) is False:
        print("at least one of scoring methods should be included: either --score_based or --embedding_based ")
        exit()

    model_output = 'ids'




    res_dict ={}

    print('#####')
    print(args.prediction_path)
    print('#####')
    res_dict[args.prediction_path] = {}
    for task_name in all_tasks:
       
        print(f"\n{task_name}\n")
        if args.score_based:
            prf1 = get_common_PRF1ACC(task_name,args.prediction_path,model_output,no_em=False)

        if args.embedding_based:
            print("EMB:",end='\t')
            prf1_emb = get_emb_PRF1(task_name,args.prediction_path,model_output,emb_layers=['decoder4mean','encoder12mean'])
        if args.get_hit_mrr:
            print("This would take time, and require deeponto")
            emb_a = get_embeddings_score(task_name,args.prediction_path,emb_layers=['decoder4mean','encoder12mean']) #'encoder12max','decoder4max'
            print(emb_a)
        print()
