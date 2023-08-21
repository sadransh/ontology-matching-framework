#This script generates and saves embeddings for last layers of encoder and decoder into a pkl
#file. 
#needs refactor to run on multi gpu; currently one can run multiple instances of same script on multiple checkpoint
# command to run:
# python embedding_generation_gpu.py --task_name[body/phar/neop] --is_target[0/1] --gpu_num[0-7] tempurature_scaling_value[1.0 to any] path_to_checkpoints_folder


#import
import json
import os
import timeit
import re
from datetime import datetime
from typing import List
import pickle
import argparse
import pytz
from dataclasses import dataclass

import torch
from torch import nn
from transformers import set_seed,T5ForConditionalGeneration, AutoTokenizer
import numpy as np
import tqdm
import inflect


today = datetime.now(pytz.timezone("US/Pacific")).strftime("%m-%d-%y")
sing_eng = inflect.engine()

#dictionary for numerical conversions in data
rep_dict_num={
    'first':'1',
    'second':'2',
    'third':'3',
    'forth':'4',
    'fifth':'5',
    'sixth':'6',
    'seventh':'7',
    'eighth':'8',
    'ninth':'9',
    'tenth':'10',
    'eleventh':'11',
    'twelfth':'12'
}

@dataclass
class PredictedValue:
    value: str
    probability: float=1
    


@dataclass
class NormalizedSnomedCT:
    code: List[PredictedValue]
    #fsn: PredictedValue
    probability: float
        
@dataclass
class NCITtermFSN:
    diagnosis_name: str
    def serialize(self) -> str:
        return f"F0: {self.diagnosis_name.strip()} F2:" 

@dataclass
class FMAtermFSN:
    diagnosis_name: str
    def serialize(self) -> str:
        return f"F0: {self.diagnosis_name.strip()} F1:"    

@dataclass
class NCITtermSYN:
    diagnosis_name: str
    def serialize(self) -> str:
        return f"S0: {self.diagnosis_name.strip()} F2:" 

@dataclass
class FMAtermSYN:
    diagnosis_name: str
    def serialize(self) -> str:
        return f"S0: {self.diagnosis_name.strip()} F1:"  

@dataclass
class FMAIDNormalize:
    diagnosis_name: str
    def serialize(self) -> str:
        return f"F0: {self.diagnosis_name.strip()} 1-" 

@dataclass
class FMAIDNormalizeSYN:
    diagnosis_name: str
    def serialize(self) -> str:
        # return f"S0: {self.diagnosis_name.strip()} 1-"
        return f"F0: {self.diagnosis_name.strip()} 1-"

@dataclass
class NCITIDNormalize:
    diagnosis_name: str
    def serialize(self) -> str:
        return f"F0: {self.diagnosis_name.strip()} 2-" 

@dataclass
class NCITDNormalizeSYN:
    diagnosis_name: str
    def serialize(self) -> str:
        # return f"S0: {self.diagnosis_name.strip()} 2-"
        return f"F0: {self.diagnosis_name.strip()} 2-"


class Normalizer:
    def __init__(self, model_name="",gpu_num=0):

        self.model_name = model_name
        set_seed(43)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.decoder_pool = 'mean'
        self.encoder_pool = 'max'
        self.encoder_layer = 12
        self.decoder_layer = 4
        self.gpu_num=gpu_num
        
        if torch.cuda.is_available():
            target_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
            self.device = target_devices[self.gpu_num] if self.gpu_num < len(target_devices) else target_devices[0]
        else:
            self.device = 'cpu'
        self.model.to(self.device)


    def _predict(self, terms):
        batch_size = 8
        all_batch_outputs = []
        length_sorted_idx = np.argsort([-len(term) for term in terms]) 
        terms_sorted = [terms[idx] for idx in length_sorted_idx]
        
        for i in range(0, len(terms), batch_size):
            term_batch = terms_sorted[i:i+batch_size]
            batch_output = self._encode_decode(term_batch)
            batch_output_list =[]
            batch_output_list.append(batch_output)
            all_batch_outputs.extend(batch_output_list)
        
        all_batch_outputs = [all_batch_outputs[idx] for idx in np.argsort(length_sorted_idx)]

        return all_batch_outputs


    def _embedding_pool(self, embeddings, mask, pool_type):
        input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
        # inspired by https://github.com/UKPLab/sentence-transformers/blob/d5b011583b8d689591d52b0d244be315f2800d30/sentence_transformers/models/Pooling.py#L70
        if pool_type == 'max':
            embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            pooled_embeddings = torch.max(embeddings, 1).values
        elif pool_type == 'mean':
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_embeddings = sum_embeddings / sum_mask
        return pooled_embeddings


    def _encode_decode(self, terms):

        model_inputs = self.tokenizer(terms, max_length=512 * 2, padding=True, truncation=True, return_tensors='pt')
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                max_length=512* 2,
                num_beams=beam_number,
                num_return_sequences=beam_number,
                output_scores=True,
                output_hidden_states=True
            )

        
        predicted_concepts = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

        probabilities = [[np.max(nn.Softmax(dim=-1)(s[i]/tempurature).cpu().numpy()) for s in out.scores] for i in range(len(terms))]
        all_encoder_layer_pooled = []
        all_decoder_layer_pooled = []
        for encoder_layer_num in range(12,13): #last encoder layer only to save space  change to (0,13) for more layers
            for pooling_meth in ['max','mean']:
                encoder_embeddings = out.encoder_hidden_states[encoder_layer_num]
                pooled_encoder_embeddings = self._embedding_pool(encoder_embeddings, attention_mask,pooling_meth)
                all_encoder_layer_pooled.append(pooled_encoder_embeddings.cpu().numpy())

        for decoder_layer_num in range(4,5): #last decoder layer only to save space change to (0,5) for more layers
            for pooling_meth in ['max','mean']:
                output_mask = out.sequences.clone()
                output_mask = output_mask[:,1:]
                output_mask[output_mask>0] = 1

                choice_idx = 0 # for greedy search we have just one choice
                decoder_embeddings = torch.stack([torch.stack([out.decoder_hidden_states[char_idx][decoder_layer_num][term_idx][choice_idx] \
                    for char_idx in range(out.sequences.shape[1] - 1)]) for term_idx in range(out.sequences.shape[0])])

                pooled_decoder_embeddings = self._embedding_pool(decoder_embeddings, output_mask,pooling_meth)
                all_decoder_layer_pooled.append(pooled_decoder_embeddings.cpu().numpy())

        
        return list(zip(predicted_concepts, probabilities, [all_encoder_layer_pooled], [all_decoder_layer_pooled]))
    

    def normalize(self, terms):
        terms_str = [t.serialize() for t in terms]
        prediction_output = self._predict(terms_str)
        decoder_output_str = [prediction_output[0][i][0] for i in range(beam_number)]
        char_prob = [prediction_output[0][i][1] for i in range(beam_number)]
        decoder_output = [self.parse_decoder_output(decoder_output_str_i, char_prob_i) for decoder_output_str_i, char_prob_i in zip(decoder_output_str,char_prob)]
        return decoder_output, [embedding for _, _, embedding, _ in prediction_output[0]], [embedding for _, _, _, embedding in prediction_output[0]]

    def _parse_component(self, component_str, char_probs, delimiter, count=None):

        part_probs = []
        char_idx = 0

        parts = component_str.split(delimiter)
        delimiter_len = len(delimiter)
        for part in parts:
            part_probs.append(np.prod(char_probs[char_idx: char_idx + len(part) ])) # + delimiter_len do not include the probability of delimiter_len
            char_idx += len(part) + len(delimiter)
        if count is not None:
            while len(parts) < count:
                print(f"adding fake empty parts to meet component count requirement of {count}, parts: {parts}")
                parts.append('')
                part_probs.append(0)
            if len(parts) > count:
                print(f"trimming parts to meet component count requirement of {count}, parts: {parts}")
                parts = parts[:count]
                part_probs = part_probs[:count]
                part_probs[-1] = 0 # force the probability of last component to zero
        return parts, part_probs
    

    def parse_decoder_output(self, decoder_output_str, char_probs):
        char_probs.append(1.0) # add a fake probability as the delimiter's len is 2 
        snomed_ct_strs, snomed_ct_probs = self._parse_component(decoder_output_str, char_probs, '][')

        return NormalizedOnt([PredictedValue(c, p) for c, p in zip(snomed_ct_strs, snomed_ct_probs)],
                                np.prod(snomed_ct_probs) )

class conceptNormalizer(Normalizer):
    def __init__(self, model_name,gpu_num):
        super().__init__(model_name = model_name,gpu_num=gpu_num)
    def parse_decoder_output(self, decoder_output_str, char_probs):
        char_probs.append(1.0) # add a fake probability as the delimiter's len is 2 
        snomed_ct_strs, snomed_ct_probs = self._parse_component(decoder_output_str, char_probs, '][')
        return NormalizedSnomedCT([PredictedValue(c, p) for c, p in zip(snomed_ct_strs, snomed_ct_probs)],
                                np.prod(snomed_ct_probs) )

def flatten(seq):
    return [val for sublist in seq for val in sublist]

def extract_text(fully_specified_name):
    """This function extracts the SMOMED name given a SNOMED CT definitions.
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

# Load all definition files - paper verison:
def resources(task):
    DATA_FOLDER = '../data/ontology_json'
    # if task in ['body','phar','neop']:
    if task == 'body': 
        with open(os.path.join(f"{DATA_FOLDER}",'snomed_body_concepts.json'), 'r') as f:
            source_concepts = json.load(f)

        with open(os.path.join(f"{DATA_FOLDER}",'fma_concepts_body.json'), 'r') as f:
            tgt_concepts = json.load(f)
        test_data = list(source_concepts.keys())

            
    if task == 'phar': 
        with open(os.path.join(f"{DATA_FOLDER}/",'snomed_phar_concepts.json'), 'r') as f:
            source_concepts = json.load(f)
        with open(os.path.join(f"{DATA_FOLDER}/",'ncit_phar_concepts.json'), 'r') as f:
            tgt_concepts = json.load(f)
        test_data = list(source_concepts.keys())

    if task == 'neop': 
        with open(os.path.join(f"{DATA_FOLDER}/",'snomed_neop_concepts.json'), 'r') as f:
            source_concepts = json.load(f)
        with open(os.path.join(f"{DATA_FOLDER}/",'ncit_neop_concepts.json'), 'r') as f:
            tgt_concepts = json.load(f)
        test_data = list(source_concepts.keys())
    
    ###updating labels based on older ontologies: 
    with open(f"{DATA_FOLDER}/snomed_ct_concepts20160901.json", "r") as fp:
        old_snomed_concepts = json.load(fp)
    def extract_text(fully_specified_name):
        return " ".join([re.sub(r"\([^)]*\)", "", fully_specified_name).rstrip()]+re.findall(r'\([^)]*\)', fully_specified_name)[:-1]).rstrip()
    for k,v in source_concepts.items():
        if k in old_snomed_concepts.keys():
            old_values = old_snomed_concepts.get(k,'None')
            if old_values !='None' and extract_text(v['FullySpecifiedName']) != extract_text(old_values['FullySpecifiedName']):
                v['Synonyms'].append(extract_text(old_values['FullySpecifiedName']))

   
    inp_test_all = []
    for i in test_data:
        inp_test_all.append({
            'input':source_concepts.get(i)["FullySpecifiedName"] if source_concepts.get(i) else '',
            'input_snomed_code' : i,
        })

    return source_concepts,tgt_concepts,inp_test_all


def get_valid_codes(tgt_concepts):
    
    # fsn or labels to true codes : changed to lower case...
    fsn_to_id = {}
    for code,val in tgt_concepts.items():
        descriptions = val['Preferred_name'] + val['Label'] + val['Synonym']

        descriptions = [i.lower() for i in list(set([i for i in descriptions if i!='']))]
        for d in descriptions:
            if fsn_to_id.get(d):
                fsn_to_id[d] = fsn_to_id[d] + [code]
            else:
                fsn_to_id[d] = [code]
    
    multiple_codes_same_desc = {key:val for key,val in \
                            fsn_to_id.items() if len(val)>1}
    
    print(f"No of descriptions associated with multiple codes : {len(multiple_codes_same_desc)}")
    
    ### hierarchical ids to true codes 
    with open(os.path.join("../data/",'omversion_fma_to_hierarchicalId.json'), 'r', encoding='UTF-8') as f:
        fma_tids = json.load(f)
    with open(os.path.join("../data/",'omversion_ncit_to_hierarchicalId.json'), 'r', encoding='UTF-8') as f:
        ncit_tids = json.load(f)

    dic_comb = {**fma_tids, **ncit_tids}
    ids_code = {val:key for key,val in dic_comb.items()}
    
    return fsn_to_id,ids_code



def make_singular(inp):
    res = inp.split(" ")
    res2 = []
    for i in res:
        if len(i)>1 and sing_eng.singular_noun(i):
            res2.append(sing_eng.singular_noun(i))
        else:
            res2.append(i)
    return " ".join(res2)

def evaluate_model(task,model_output,k = 10,subset_for_testing = False):

    start = timeit.default_timer()
    
    all_predicted_codes_fsn = []
    all_predicted_codes = []
    all_predicted_codes_prob = []
    encoder_data = []
    decoder_data = []
    all_inputs = []
    all_inputs_fsn_syn=[]

    source_concepts,tgt_concepts,eval_data = resources(task)
    fsn_to_id,ids_code = get_valid_codes(tgt_concepts)
    
    if subset_for_testing:
        eval_data = eval_data[0:k]
    

    for i, data_dict in enumerate(tqdm.tqdm(eval_data)):   
        d = extract_text(data_dict['input'])
        #syn


        d2 = list(set(source_concepts[data_dict['input_snomed_code'].split(':')[-1]]['Synonyms']))      #+[data_dict['input']]   
        
        aux_terms,replaced_terms =[],[]

            
        for term in [d]+d2+replaced_terms:
            #senario 1 - making singual form of terms and adding up as synonym of the class
            sig_d = make_singular(term)
            if term !=sig_d:
                aux_terms.append(sig_d)
            
        d2 = list(set(aux_terms+d2+replaced_terms)-set(d))


        #def
        d3 = list(set(source_concepts[data_dict['input_snomed_code']\
                    .split(':')[-1]]['Definitions'])) 
        d4 = list(set(source_concepts[data_dict['input_snomed_code'].split(':')[-1]]['Parents'])) + list(set(source_concepts[data_dict['input_snomed_code'].split(':')[-1]]['Children'])) 
        if model_output == 'label': 
            ## label to label
            if task=='body':
                fsn_input = FMAtermFSN(d)
            else:
                fsn_input = NCITtermFSN(d)

            aa = normalizerD.normalize([fsn_input]) 
            f_predcodes = [[i.value for i in j.code] for j in aa]
            f_predprob = [[i.probability for i in j.code] for j in aa]
            # saving fsn output separately
            all_predicted_codes_fsn.append((f_predcodes,f_predprob))

            # removing invalid codes in the fsn to be added with synonymns prediction
            pred = [(i[0],j[0]) for i,j in zip(f_predcodes,f_predprob) \
                       if fsn_to_id.get(i[0].lower()) and i[0]!='']
            for k in d2:
                if task=='body':
                    syn_input = FMAtermSYN(k)
                else:
                    syn_input = NCITtermSYN(k)
                aa = normalizerD.normalize([syn_input])
                s_predcodes = [[i.value for i in j.code] for j in aa]
                s_predprob = [[i.probability for i in j.code] for j in aa]
                pred += [(i[0],j[0]) for i,j in zip(s_predcodes,s_predprob)\
                            if fsn_to_id.get(i[0].lower()) and i[0]!='']
                   
        elif model_output == 'ids': 
            
            if task=='body':
                fsn_input = FMAIDNormalize(d)
                prompt = '1-'
            else:
                fsn_input = NCITIDNormalize(d)
                prompt = '2-'

            aa,enc_emb,dec_emb = normalizerD.normalize([fsn_input]) 
            f_predcodes = [[i.value for i in j.code] for j in aa]
            f_predprob = [[i.probability for i in j.code] for j in aa]
            encoder_data.append(enc_emb)
            decoder_data.append(dec_emb)
            all_predicted_codes_fsn.append((f_predcodes,f_predprob))
            all_inputs.append(data_dict['input_snomed_code'])
            all_inputs_fsn_syn.append(d)

            pred = [(i[0],j[0]) for i,j in zip(f_predcodes,f_predprob) \
                       if ids_code.get(prompt+i[0]) and i[0]!='']
            
            for k in d2:

                if task=='body':
                    syn_input = FMAIDNormalizeSYN(k)
                else:
                    syn_input = NCITDNormalizeSYN(k)
                aa,enc_syn,dec_syn = normalizerD.normalize([syn_input])
                encoder_data.append(enc_syn)
                decoder_data.append(dec_syn)
                all_inputs.append("Syn:"+data_dict['input_snomed_code'])
                all_inputs_fsn_syn.append(k)

                # print("aa",aa)
                s_predcodes = [[i.value for i in j.code] for j in aa]
                s_predprob = [[i.probability for i in j.code] for j in aa]
                pred += [(i[0],j[0]) for i,j in zip(s_predcodes,s_predprob)\
                            if ids_code.get(prompt+i[0]) and i[0]!='']
            for k in d3:

                if task=='body':
                    syn_input = FMAIDNormalizeSYN(k)
                else:
                    syn_input = NCITDNormalizeSYN(k)
                aa,enc_syn,dec_syn = normalizerD.normalize([syn_input])
                encoder_data.append(enc_syn)
                decoder_data.append(dec_syn)
                all_inputs.append("Def:"+data_dict['input_snomed_code'])
                all_inputs_fsn_syn.append(k)

                s_predcodes = [[i.value for i in j.code] for j in aa]
                s_predprob = [[i.probability for i in j.code] for j in aa]
                pred += [(i[0],j[0]) for i,j in zip(s_predcodes,s_predprob)\
                            if ids_code.get(prompt+i[0]) and i[0]!='']
        

        raw_codes = [i[0] for i in pred]
        raw_scores = [i[1] for i in pred]
       
    
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


    stop = timeit.default_timer()

    print('Time: ', stop - start, )  

    encoder_data = np.stack(encoder_data)
    decoder_data = np.stack(decoder_data)
    return all_inputs_fsn_syn,all_inputs,all_predicted_codes, all_predicted_codes_prob, all_predicted_codes_fsn,encoder_data,decoder_data

def get_target_embeddings(task,model_output,k = 10,subset_for_testing = False):

    start = timeit.default_timer()
    
    all_predicted_codes_fsn = []
    all_predicted_codes = []
    all_predicted_codes_prob = []
    encoder_data = []
    decoder_data = []
    all_inputs = []
    all_inputs_fsn_syn=[]

    source_concepts,tgt_concepts,eval_data = resources(task)
    fsn_to_id,ids_code = get_valid_codes(tgt_concepts)
    
    target_concepts = []
    for key,val in tgt_concepts.items():
        for l in   tgt_concepts.get(key)["Label"]:
            target_concepts.append({
                'input':l if tgt_concepts.get(key) else '',
                'input_code' : key,
            })

    if subset_for_testing:
        target_concepts = target_concepts[0:k]
    

    for i, data_dict in enumerate(tqdm.tqdm(target_concepts)): 

        d = data_dict['input']
        #syn
        d2 = list(set(tgt_concepts[data_dict['input_code']\
                    .split(':')[-1]]['Synonym']))                
        aux_terms,replaced_terms =[],[]


        #doing some inference time fixure on inputs to boost performance
        for term in [d]+d2+replaced_terms:
        #senario 1 - adding a singular synonym of terms to class
            sig_d = make_singular(term)
            if term !=sig_d:
                aux_terms.append(sig_d)
        
        d2 = list(set(aux_terms+d2+replaced_terms)-set(d))

        aux_terms_new= []
        ordinal_copy = d2
        # senario 2 - coverting ordinal numbers to numerical form for better matching
        capt = [i.capitalize() for i in list(rep_dict_num.keys())]
        for item in list(rep_dict_num.keys())+capt:
            for term in ordinal_copy:
                if item in term:
                    term_new = term.replace(item,rep_dict_num[item.lower()])
                
                    if term !=term_new:
                        aux_terms_new.append(term_new)
        
        d2 = list(set(d2+aux_terms_new))


        #def
        d3 = list(set(tgt_concepts[data_dict['input_code']\
                    .split(':')[-1]]['Definition'])) 
        d4 = list(set(tgt_concepts[data_dict['input_code']\
                    .split(':')[-1]]['Preferred_name'])) 
                   
        if model_output == 'ids': 
            
            if task=='body':
                fsn_input = FMAIDNormalize(d)
                prompt = '1-'
            else:
                fsn_input = NCITIDNormalize(d)
                prompt = '2-'
            aa,enc_emb,dec_emb = normalizerD.normalize([fsn_input]) 
            f_predcodes = [[i.value for i in j.code] for j in aa]
            f_predprob = [[i.probability for i in j.code] for j in aa]
            encoder_data.append(enc_emb)
            decoder_data.append(dec_emb)
            all_predicted_codes_fsn.append((f_predcodes,f_predprob))
            all_inputs.append(data_dict['input_code'])
            all_inputs_fsn_syn.append(d)

            ### Adding 'Syn to ids' in the 'Fsn to ids'
            
            # Removing invalid codes from 'Fsn to ids'
            pred = [(i[0],j[0]) for i,j in zip(f_predcodes,f_predprob) \
                       if ids_code.get(prompt+i[0]) and i[0]!='']
            
            for k in d2:

                if task=='body':
                    syn_input = FMAIDNormalizeSYN(k)
                else:
                    syn_input = NCITDNormalizeSYN(k)
                aa,enc_syn,dec_syn = normalizerD.normalize([syn_input])
                encoder_data.append(enc_syn)
                decoder_data.append(dec_syn)
                all_inputs.append("Syn:"+data_dict['input_code'])
                all_inputs_fsn_syn.append(k)


                s_predcodes = [[i.value for i in j.code] for j in aa]
                s_predprob = [[i.probability for i in j.code] for j in aa]
                pred += [(i[0],j[0]) for i,j in zip(s_predcodes,s_predprob)\
                            if ids_code.get(prompt+i[0]) and i[0]!='']
            for k in d3:

                if task=='body':
                    syn_input = FMAIDNormalizeSYN(k)
                else:
                    syn_input = NCITDNormalizeSYN(k)
                aa,enc_syn,dec_syn = normalizerD.normalize([syn_input])
                encoder_data.append(enc_syn)
                decoder_data.append(dec_syn)
                all_inputs.append("Def:"+data_dict['input_code'])
                all_inputs_fsn_syn.append(k)

                s_predcodes = [[i.value for i in j.code] for j in aa]
                s_predprob = [[i.probability for i in j.code] for j in aa]
                pred += [(i[0],j[0]) for i,j in zip(s_predcodes,s_predprob)\
                            if ids_code.get(prompt+i[0]) and i[0]!='']
            for k in d4:

                if task=='body':
                    syn_input = FMAIDNormalizeSYN(k)
                else:
                    syn_input = NCITDNormalizeSYN(k)
                aa,enc_syn,dec_syn = normalizerD.normalize([syn_input])
                encoder_data.append(enc_syn)
                decoder_data.append(dec_syn)
                all_inputs.append("pfn:"+data_dict['input_code'])
                all_inputs_fsn_syn.append(k)

                # print("aa",aa)
                s_predcodes = [[i.value for i in j.code] for j in aa]
                s_predprob = [[i.probability for i in j.code] for j in aa]
                pred += [(i[0],j[0]) for i,j in zip(s_predcodes,s_predprob)\
                            if ids_code.get(prompt+i[0]) and i[0]!='']        
        # get the combined raw codes and scores for valid codes only 
        raw_codes = [i[0] for i in pred]
        raw_scores = [i[1] for i in pred]
       
    
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


    stop = timeit.default_timer()

    print('Time: ', stop - start, np.stack(encoder_data).shape,np.stack(encoder_data).shape)  


    return all_inputs_fsn_syn,all_inputs,all_predicted_codes, all_predicted_codes_prob, all_predicted_codes_fsn,encoder_data,decoder_data



def save_files(input_keys,all_predicted_codes, all_predicted_codes_prob, all_predicted_codes_fsn,task = 'body', model_output = 'ids', checkpoint = '120000',istarget='0'):
    json_list_test_all = []

    for inp,pred_all, pred_prob_all, pred_fsn_all in zip(input_keys,all_predicted_codes, all_predicted_codes_prob, all_predicted_codes_fsn):


        new_fsn_all = (pred_fsn_all[0], [[float(p) for p in prob] for prob in pred_fsn_all[1]])

        json_list_test_all.append({'pred_c1': pred_all,
                                'pred_prob_all': [float(p) for p in pred_prob_all],
                                'pred_fsn_all': new_fsn_all,
                                'input_code':inp,
                                })  


    json_object = json.dumps(json_list_test_all)
    is_target_in_name = '1' if istarget else '0'
    # Writing to sample.json
    with open(f'../data/results-{today}-checkpoint-{checkpoint}/results-{checkpoint}-{task}-{model_output}-{is_target_in_name}.json', 'w', encoding='utf-8') as outfile:
        outfile.write(json_object)




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="embedding generation script")

    # Add arguments
    parser.add_argument("--task_name", type=str, default="", help="name of the task",required=True)
    parser.add_argument("--is_target",  help="if the embeddings should generated for target",action='store_true')
    parser.add_argument("--gpu_num", type=int, default=0, help="gpu num if the device is multi GPU",required=True)
    parser.add_argument("--temp_scale", type=float, default=2.0, help="gpu num if the device is multi GPU",required=False)
    parser.add_argument("--checkpoint_path", type=str, default="", help="path_to_checkpoint",required=True)


    # Parse the arguments
    args = parser.parse_args()
  
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)
    GPU_DEVICE = args.gpu_num
    path_to_checkpoint =args.checkpoint_path if args.checkpoint_path[-1]!='/' else args.checkpoint_path[:-1]
    beam_number = 1

    tempurature = args.temp_scale
    checknum = args.checkpoint_path.split("/")[-1].split("checkpoint-")[1]



    model_output = 'ids'

    if not os.path.exists(f'../data/results-{today}-checkpoint-{checknum}'):
        os.mkdir(f'../data/results-{today}-checkpoint-{checknum}')



    normalizerD = conceptNormalizer(path_to_checkpoint,gpu_num=GPU_DEVICE)

    print (f"------------------------- Checkpoint: {checknum} ---------------------------")

    print (f"------------------------- Task: {args.task_name}, Output: {model_output}, GPU: {GPU_DEVICE}, Is Target:{args.is_target} ------------------------- ")

    
    if args.is_target:
        all_inputs_fsn,input_concept_code,pred_c1, pred_prob_c1, fsn_pred_c1,encoder_embs,decoder_embs = get_target_embeddings(task = args.task_name, model_output = model_output, k = 100, subset_for_testing = False)
    else:
        all_inputs_fsn,input_concept_code,pred_c1, pred_prob_c1, fsn_pred_c1,encoder_embs,decoder_embs = evaluate_model(task = args.task_name, model_output = model_output, k = 100, subset_for_testing = False)

    
    
    #loop for properly saving embeddings into quantized_embeddings
    i=0
    quantized_embeddings = {}
    print("length",len(all_inputs_fsn),len(input_concept_code),len(encoder_embs),len(decoder_embs),len(pred_c1),len(pred_prob_c1),len(fsn_pred_c1))
    for fsn,concept_code,encoder,decoder in zip(all_inputs_fsn,input_concept_code,encoder_embs,decoder_embs):
        encoder = encoder[0]
        decoder = decoder[0]
        tmp=concept_code.split(":")
        i+=1
        if len(tmp)==1: #handles parent
            quantized_embeddings[concept_code] = {
            'concept': concept_code,
            'fsn':fsn,
            'Syn':[], #will be filed later with else
            'Def':[], #will be filed later with else
            }
            quantized_embeddings[concept_code].update({f'encoder12{str(j)}':encoder[i] for i,j in zip(range(len(encoder)),['max','mean'])})
            quantized_embeddings[concept_code].update({f'decoder4{str(j)}':decoder[i] for i,j in zip(range(len(decoder)),['max','mean'])})

        else : #handles (syn/def)
            subkey=tmp[0] #either Def/Syn

            deflist = quantized_embeddings[tmp[1]].get(subkey,[])
            tmp_dict = {
            'concept': tmp[1], 
            'fsn':fsn,
            }
            tmp_dict.update({f'encoder12{str(j)}':encoder[i] for i,j in zip(range(len(encoder)),['max','mean'])})
            tmp_dict.update({f'decoder4{str(j)}':decoder[i] for i,j in zip(range(len(decoder)),['max','mean'])})
            deflist.append(tmp_dict)
            quantized_embeddings[tmp[1]][subkey] = deflist

    print("length of data i=",i,len(set(list(quantized_embeddings.keys()))))
        

    #saves embeddings for each concept code
    is_target_in_name = 'tgt-' if args.is_target else ''
    
    with open(f'../data/results-{today}-checkpoint-{checknum}/embeddings-{is_target_in_name}{checknum}-{model_output}-{args.task_name}.pkl', 'wb') as f:
        pickle.dump(quantized_embeddings, f)

    #saves raw prediction and scores
    save_files(input_concept_code,pred_c1, pred_prob_c1, fsn_pred_c1, task = args.task_name, model_output = model_output, checkpoint = checknum,istarget=args.is_target)

    print(f"saved under ../data/results-{today}-checkpoint-{checknum}")
