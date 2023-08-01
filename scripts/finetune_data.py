import os
import re
import json
from tqdm.auto import tqdm
from collections import defaultdict
import datetime
import os
import random
from pathlib import Path


today = datetime.datetime.now().strftime("%m-%d-%y")

DATA_DIR = '../data'

print("loading subset ontologies ...")

data_dict = defaultdict()
for file_name in os.listdir(f"{DATA_DIR}/ontology_json/"):
    data_name , file_format = file_name.split('.')
    data_name = data_name.replace("_concepts","")
    if file_format == 'json' and any(i in data_name for i in ['body','phar','neop']):
        with open(f"{DATA_DIR}/ontology_json/{file_name}", "r") as fp:
            data_dict[data_name] = list(json.load(fp).keys())
            print(f"loaded {data_name} with len {len(data_dict[data_name])}")



# PAPAER VERSION:
with open(f"{DATA_DIR}/ontology_json/snomed_concepts_all.json", "r") as fp:
    snomed_concepts = json.load(fp)

with open(f"{DATA_DIR}/ontology_json/fma_concepts_all.json", "r") as fp:
    fma_concepts = json.load(fp)

with open(f"{DATA_DIR}/ontology_json/ncit_concepts_all.json", "r") as fp:
    ncit_concepts = json.load(fp)
 
## h-IDS:

with open(f"{DATA_DIR}/omversion_fma_to_hierarchicalId.json", "r") as fp:
    fma_to_treeid = json.load(fp)
treeid_to_fma = { v:k for k,v in fma_to_treeid.items()}

with open(f"{DATA_DIR}/omversion_ncit_to_hierarchicalId.json", "r") as fp:
    ncit_to_treeid = json.load(fp)
treeid_to_ncit = { v:k for k,v in ncit_to_treeid.items()}

with open(f"{DATA_DIR}/omversion_snomed_to_hierarchicalId.json", "r") as fp:
    sid_to_treeid = json.load(fp)
treeid_id_to_sid = { v:k for k,v in sid_to_treeid.items()}



def extract_text_extend(fully_specified_name):
    return " ".join([re.sub(r"\([^)]*\)", "", fully_specified_name).rstrip()]+re.findall(r'\([^)]*\)', fully_specified_name)[:-1]).rstrip()


print("concept extension ... ")

###extending concepts based on older snomed release: 
with open(f"../data/ontology_json/snomed_ct_concepts20160901.json", "r") as fp:
    old_snomed_concepts = json.load(fp)

for k,v in snomed_concepts.items():
    if k in old_snomed_concepts.keys():
        old_values = old_snomed_concepts.get(k,'None')
        if old_values !='None' and extract_text_extend(v['FullySpecifiedName']) != extract_text_extend(old_values['FullySpecifiedName']) :
            #print(extract_text_extend(old_values['FullySpecifiedName']),"->" ,v['FullySpecifiedName'])
            v['Synonyms'].append(extract_text_extend(old_values['FullySpecifiedName']))



def fmatreeid_to_fsn(tree_id):
    return fmaid_to_fsn[treeid_to_fma[tree_id]]
def ncittreeid_to_fsn(tree_id):
    return ncitid_to_fsn[treeid_to_ncit[tree_id]]


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


def snomed_to_id_subset(data_type , lower_case= False):
    Snomed_to_id = defaultdict(list)
    non_existing_sids = []

    if data_type == "FMA":
        dataset = data_dict['snomed_body']
    elif data_type == "NEOP":
        dataset = data_dict['snomed_neop']
    elif data_type == "PHAR":
        dataset = data_dict['snomed_phar']
    elif data_type == "ALL":
        dataset = snomed_concepts.keys()
    else:
        print("please choose between FMA, NEOP, PHA, or change subset_only to False")
        
    for sid in dataset:

        try:
            concept_detail = snomed_concepts[sid] 
        except:
            non_existing_sids.append(sid)
            continue

        FSN = extract_text(concept_detail['FullySpecifiedName']).lower() if lower_case else extract_text(concept_detail['FullySpecifiedName'])
        Snomed_to_id[FSN].append(sid)

        for syn in concept_detail['Synonyms']:
            syn = syn.lower() if lower_case else syn
            Snomed_to_id[syn].append(sid)

        for definition in concept_detail['Definitions']:
            definition = definition.lower() if lower_case else definition
            Snomed_to_id[definition].append(sid)
    
    return Snomed_to_id
    

Snomed_to_id = snomed_to_id_subset("FMA", lower_case =True)


def create_FMA_pretraining( dataset , output_id = False , subset_only = True , lower_case = False):

    non_existing_fmaids = []
    fma_matching = dict()
    fma_finetuning_train = []
    
    Snomed_to_id = snomed_to_id_subset("FMA", lower_case=lower_case) if subset_only else snomed_to_id_subset("ALL", lower_case=lower_case)

    # FOR FMA:
    Synonyms = "Synonym"
    FullySpecifiedName = "Label"
    Definitions = "Definition" 

    
    def create_snomed_datapoints(SNOMED_IDs):
        datapoints = []
        # SNOMED to FMA:

        for sid in SNOMED_IDs:
            snomed_concept  = snomed_concepts[sid] 

            datapoints.append(
                {'input': snomed_concept["FullySpecifiedName"],
                'output': output,
                'record_type': f'FMA{record_type}:F0-F1',
                'id': f'FMA_BODY-{fmaid}',
                'snomed_id' : sid })
            for syn in snomed_concept['Synonyms']:
                if syn:# and syn[:6]!='Entire':
                    datapoints.append(
                        {'input': syn,
                         'output': output,
                         'record_type': f'FMA{record_type}:S0-F1',
                         'id': f'FMA_BODY-{fmaid}',
                         'snomed_id' : sid })
            for definition in snomed_concept['Definitions']:
                if definition:# and definition[:6]!='Entire':
                    datapoints.append(
                        {'input': definition,
                         'output': output,
                         'record_type': f'FMA{record_type}:S0-F1',
                         'id': f'FMA_BODY-{fmaid}',
                         'snomed_id' : sid })
        return datapoints 


    def create_fma_to_fma():
        datapoints = []
        for syn in concept_detail[Synonyms]:
            if syn:# and syn[:6]!='Entire':
                datapoints.append(
                    {'input': syn,
                     'output': output,
                     'record_type': f'FMA{record_type}:S1-F1',
                     'id': f'FMA_BODY-{fmaid}',
                     'snomed_id' : False })
        
        # FOR FSN to TREE:
        if output_id:
            if FSN:
                datapoints.append(
                    {'input': FSN,
                     'output': output,
                     'record_type': f'FMA{record_type}:F1-F1',
                     'id': f'FMA_BODY-{fmaid}',
                     'snomed_id' : False })
            
        return datapoints



    for fmaid in tqdm(dataset):
        datapoint = defaultdict(list)
        try:
            concept_detail = fma_concepts[fmaid] 
        except:
            non_existing_fmaids.append(fmaid)
            continue

        datapoint["matched_snomed"] = False
        datapoint["fma_id"] = fmaid

        FSN = concept_detail[FullySpecifiedName][0] 
        
        if output_id:
            output = fma_to_treeid[fmaid]
            record_type = "_tree"
        else:
            output = FSN
            record_type = ''
        
        FSN = FSN.lower() if lower_case else FSN
        
        if Snomed_to_id.get(FSN):
            datapoint["matched_snomed"] = True
            SNOMED_IDs = Snomed_to_id[FSN]
            datapoint["SNOMED_ID"].append(SNOMED_IDs)

            # SNOMED to FMA:
            datapoins = create_snomed_datapoints(SNOMED_IDs)
            fma_finetuning_train = fma_finetuning_train + datapoins

            # REST: FMA TO FMA:
            datapoins = create_fma_to_fma()
            fma_finetuning_train = fma_finetuning_train + datapoins



        for syn in concept_detail[Synonyms]:
            if syn:
                syn =  syn.lower() if lower_case else syn
                if Snomed_to_id.get(syn):
                    datapoint["matched_snomed"] = True
                    SNOMED_IDs = Snomed_to_id[syn]
                    datapoint["SNOMED_ID"].append(SNOMED_IDs)

                    # SNOMED to FMA:
                    datapoins = create_snomed_datapoints(SNOMED_IDs)
                    fma_finetuning_train = fma_finetuning_train + datapoins


        if  datapoint["matched_snomed"] == False:
            # REST: FMA TO FMA:
            datapoins = create_fma_to_fma()
            fma_finetuning_train = fma_finetuning_train + datapoins


        fma_matching[fmaid] = datapoint
    
    
    matched = 0
    matched_morethanonce = 0
    for k,v in fma_matching.items():
        if v['matched_snomed']:
            matched +=1
            if len(v['SNOMED_ID']) > 1:
                # print(v['fma_id'])
                # print(v['SNOMED_ID'])
                matched_morethanonce += 1
    
    snomed_counter = 0
    for data in fma_finetuning_train:
        if data['snomed_id']:
            snomed_counter+=1

    print(f"Number of matched with muliple SNOMEDS: : {matched_morethanonce}")
    print(f"Number of matched IDS: {matched}")
    print(f"Perceny of the IDs: {round((100* matched )/ len(dataset),2)}%")
    print(f"Out of the total of {len(fma_finetuning_train)} train data, {snomed_counter} are matched from SNOMED")

    return fma_finetuning_train, fma_matching
    

print("generating fma data ...")
fma_finetuning_train_tree, fma_matching_tree = create_FMA_pretraining(data_dict['fma_body'],output_id = True, subset_only = True, lower_case= True)

fma_finetuning_train, fma_matching = create_FMA_pretraining(data_dict['fma_body'],output_id = False, subset_only = True , lower_case=True)


# NCIT:
def create_NCIT_pretraining( dataset , NCIT_TYPE = "PHAR" , output_id = False , subset_only = True , lower_case= False):

    Synonyms = "Synonym"
    # FullySpecifiedName = "DisplayName"
    FullySpecifiedName = "Label"
    
    non_existing_ncitids = []
    ncit_pharm_matching = dict()
    ncit_pharm_finetuning_train = []
    
    Snomed_to_id = snomed_to_id_subset(NCIT_TYPE , lower_case = lower_case) if subset_only else snomed_to_id_subset("ALL", lower_case= lower_case)


    def create_snomed_datapoints_NCIT(SNOMED_IDs ):
        datapoints = []
        # SNOMED to NCIT:
        for sid in SNOMED_IDs:
            snomed_concept  = snomed_concepts[sid] 

            datapoints.append(
                {'input': snomed_concept["FullySpecifiedName"],
                 'output': output,
                 'record_type': f'NCIT_{NCIT_TYPE}{record_type}:F0-F2',
                 'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
                 'snomed_id' : sid })
            for syn in snomed_concept['Synonyms']:
                if syn:
                    datapoints.append(
                        {'input': syn,
                         'output': output,
                         'record_type': f'NCIT_{NCIT_TYPE}{record_type}:S0-F2',
                         'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
                         'snomed_id' : sid })
            for definition in snomed_concept['Definitions']:
                if definition:
                    datapoints.append(
                        {'input': definition,
                         'output': output,
                         'record_type': f'NCIT_{NCIT_TYPE}{record_type}:S0-F2',
                         'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
                         'snomed_id' : sid })
        return datapoints


    def create_ncit_to_ncit():
        datapoints = []
        for syn in concept_detail[Synonyms]:
            if syn:
                datapoints.append(
                    {'input': syn,
                     'output': output,
                     'record_type': f'NCIT_{NCIT_TYPE}{record_type}:S2-F2',
                     'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
                     'snomed_id' : False })
        
        # FOR FSN to TREE:
        if output_id:
            if FSN:
                datapoints.append(
                    {'input': FSN,
                     'output': output,
                     'record_type': f'NCIT_{NCIT_TYPE}{record_type}:F2-F2',
                     'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
                     'snomed_id' : False })
            
        return datapoints

    
    for ncitid in tqdm(dataset):
        datapoint = defaultdict(list)
        try:
            concept_detail = ncit_concepts[ncitid] 
        except:
            non_existing_ncitids.append(ncitid)
            continue

        datapoint["matched_snomed"] = False


        FSN = concept_detail[FullySpecifiedName][0]
        
        if output_id:
            output = ncit_to_treeid[ncitid]
            record_type = "_tree"
        else:
            output = FSN
            record_type = ''
        
        FSN = FSN.lower() if lower_case else FSN
        
        if Snomed_to_id.get(FSN):
            datapoint["matched_snomed"] = True
            SNOMED_IDs = Snomed_to_id[FSN]
            datapoint["SNOMED_ID"].append(SNOMED_IDs)

            # SNOMED to NCIT:
            datapoins = create_snomed_datapoints_NCIT(SNOMED_IDs )
            ncit_pharm_finetuning_train = ncit_pharm_finetuning_train + datapoins

            # REST: NCIT TO NCIT:
            datapoins = create_ncit_to_ncit()
            ncit_pharm_finetuning_train = ncit_pharm_finetuning_train + datapoins



        for syn in concept_detail[Synonyms]:
            if syn:
                syn = syn.lower() if lower_case else syn
                if Snomed_to_id.get(syn):
                    datapoint["matched_snomed"] = True
                    SNOMED_IDs = Snomed_to_id[syn]
                    datapoint["SNOMED_ID"].append(SNOMED_IDs)

                    # SNOMED to NCIT:
                    datapoins = create_snomed_datapoints_NCIT(SNOMED_IDs)
                    ncit_pharm_finetuning_train = ncit_pharm_finetuning_train + datapoins


        if  datapoint["matched_snomed"] == False:
            # REST: NCIT TO NCIT:
            datapoins = create_ncit_to_ncit()
            ncit_pharm_finetuning_train = ncit_pharm_finetuning_train + datapoins


        ncit_pharm_matching[ncitid] = datapoint
    
    matched = 0
    matched_morethanonce = 0
    for k,v in ncit_pharm_matching.items():
        if v['matched_snomed']:
            matched +=1
            if len(v['SNOMED_ID']) > 1:

                matched_morethanonce += 1
    
    snomed_counter = 0
    for data in ncit_pharm_finetuning_train:
        if data['snomed_id']:
            snomed_counter+=1

    print(f"Number of matched with muliple SNOMEDS: : {matched_morethanonce}")
    print(f"Number of matched IDS: {matched}")
    print(f"Perceny of the IDs: {round((100* matched )/ len(dataset),2)}%")
    print(f"Out of the total of {len(ncit_pharm_finetuning_train)} train data, {snomed_counter} are matched from SNOMED")
        
    return ncit_pharm_finetuning_train , ncit_pharm_matching


print("generating ncit-phar data ...")
ncit_pharm_finetuning_train , ncit_pharm_matching = create_NCIT_pretraining( data_dict['ncit_phar'] , NCIT_TYPE = "PHAR", lower_case= True)
ncit_pharm_finetuning_train_tree , ncit_pharm_matching_tree = create_NCIT_pretraining( data_dict['ncit_phar'] , NCIT_TYPE = "PHAR" , output_id=True, lower_case= True)

print("generating ncit-neop data ...")
ncit_neop_finetuning_train , ncit_neop_matching = create_NCIT_pretraining( data_dict['ncit_neop'] , NCIT_TYPE = "NEOP", lower_case= True)
ncit_neop_finetuning_train_tree , ncit_neop_matching_tree = create_NCIT_pretraining( data_dict['ncit_neop'] , NCIT_TYPE = "NEOP", output_id=True, lower_case= True)


all_finetuning_train = ncit_pharm_finetuning_train + ncit_neop_finetuning_train + fma_finetuning_train
all_finetuning_train_tree = ncit_pharm_finetuning_train_tree + ncit_neop_finetuning_train_tree + fma_finetuning_train_tree



random.shuffle(all_finetuning_train)
random.shuffle(all_finetuning_train_tree)



def dumpjson(output_dir, data):
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(exist_ok=True, parents=True)
    with open(output_dir, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d))
            f.write('\n')
    print(f"Saved {output_dir}")

######################### THIS BLOCK WOULD BE RESOLVED AFTER FIXES TO EVAL_DATA #########################
# om_eval = []
# with open(f"{DATA_DIR}/OM_DATA/071322_def/eval_data_PAPER_071322.jsonl", 'r', encoding='utf-8') as f:
#     for l in f:
#         om_eval.append(json.loads(l))

# # %%
# om_eval[0]

# # %%
# eval_ids_FMA = []
# eval_ids_NCIT_NEOP = []
# eval_ids_NCIT_PHAR = []
# eval_ids_FMA = defaultdict(list)
# eval_ids_NCIT_NEOP = defaultdict(list)
# eval_ids_NCIT_PHAR = defaultdict(list)

# for data in tqdm(om_eval):
#     record_type , cid = data['id'].split('-')
#     if record_type == 'FMA_BODY' :
#         Snomed_to_id = snomed_to_id_subset("FMA")
#         SNOMED_IDs = Snomed_to_id[data['input']]
#         fmaid = "fma" + cid
#         eval_ids_FMA[fmaid].append(SNOMED_IDs)
#     elif record_type == 'NCIT_PHARM' :
#         Snomed_to_id = snomed_to_id_subset("PHAR")
#         SNOMED_IDs = Snomed_to_id[data['input']]
#         eval_ids_NCIT_PHAR[cid].append(SNOMED_IDs)
#     elif record_type == 'NCIT_NEOP' :
#         Snomed_to_id = snomed_to_id_subset("NEOP")
#         SNOMED_IDs = Snomed_to_id[data['input']]
#         eval_ids_NCIT_NEOP[cid].append(SNOMED_IDs)
#     else:
#         print('NO WAY')


# # FMA:
# def flatten(seq):
#     return [val for sublist in seq for val in sublist]

# def create_FMA_pretraining_eval( dataset , output_id=False):

#     # FOR FMA:
#     Synonyms = "Synonym"
#     FullySpecifiedName = "Label"
#     Definitions = "Definition"
    
#     non_existing_fmaids = []
#     fma_finetuning_eval = []
    

#     def create_snomed_datapoints(SNOMED_IDs):
#         datapoints = []
#         # SNOMED to FMA:
#         for sid in SNOMED_IDs:
#             snomed_concept  = snomed_concepts[sid] 
#             datapoints.append(
#                 {'input': snomed_concept["FullySpecifiedName"],
#                 'output': output,
#                 'record_type': f'FMA{record_type}:F0-F1',
#                 'id': f'FMA_BODY-{fmaid}',
#                 'snomed_id' : sid })
#             for syn in snomed_concept['Synonyms']:
#                 if syn:# and syn[:6]!='Entire':
#                     datapoints.append(
#                         {'input': syn,
#                          'output': output,
#                          'record_type': f'FMA{record_type}:S0-F1',
#                          'id': f'FMA_BODY-{fmaid}',
#                          'snomed_id' : sid })
#             for definition in snomed_concept['Definitions']:
#                 if definition:# and definition[:6]!='Entire':
#                     datapoints.append(
#                         {'input': definition,
#                          'output': output,
#                          'record_type': f'FMA{record_type}:S0-F1',
#                          'id': f'FMA_BODY-{fmaid}',
#                          'snomed_id' : sid })
#         return datapoints


#     def create_fma_to_fma():
#         datapoints = []
#         for syn in concept_detail[Synonyms]:
#             if syn:
#                 datapoints.append(
#                     {'input': syn,
#                      'output': output,
#                      'record_type': f'FMA{record_type}:S1-F1',
#                      'id': f'FMA_BODY-{fmaid}',
#                      'snomed_id' : False })
        
#         # FOR FSN to TREE:
#         if output_id:
#             if FSN:
#                 datapoints.append(
#                     {'input': FSN,
#                      'output': output,
#                      'record_type': f'FMA{record_type}:F1-F1',
#                      'id': f'FMA_BODY-{fmaid}',
#                      'snomed_id' : False })
            
#         return datapoints



#     for fmaid, SNOMED_IDs in tqdm(dataset.items()):
#         SNOMED_IDs = flatten(SNOMED_IDs)
        
#         try:
#             concept_detail = fma_concepts[fmaid] 
#         except:
#             non_existing_fmaids.append(fmaid)
#             continue


#         FSN = concept_detail[FullySpecifiedName][0]
        
#         if output_id:
#             output = fma_to_treeid[fmaid]
#             record_type = "_tree"
#         else:
#             output = FSN
#             record_type = ''
        

#         # SNOMED to FMA:
#         datapoins = create_snomed_datapoints(SNOMED_IDs)
#         fma_finetuning_eval = fma_finetuning_eval + datapoins

#         # REST: FMA TO FMA:
#         datapoins = create_fma_to_fma()
#         fma_finetuning_eval = fma_finetuning_eval + datapoins

#     print("non_existing_fmaids: ", non_existing_fmaids)
#     return fma_finetuning_eval


# # NCIT:

# def create_NCIT_pretraining_eval( dataset , NCIT_TYPE = "PHAR", output_id=False ):
    
#     non_existing_ncitids = []
#     ncit_finetuning_eval = []
    
#     # FOR NCIT:
#     Synonyms = "Synonym"
#     FullySpecifiedName = "Label"
#     Definitions = "Definition"
    

#     def create_snomed_datapoints_NCIT(SNOMED_IDs ):
#         datapoints = []
#         # SNOMED to NCIT:
#         for sid in SNOMED_IDs:
#             snomed_concept  = snomed_concepts[sid] 

#             datapoints.append(
#                 {'input': snomed_concept["FullySpecifiedName"],
#                  'output': output,
#                  'record_type': f'NCIT_{NCIT_TYPE}{record_type}:F0-F2',
#                  'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
#                  'snomed_id' : sid })
#             for syn in snomed_concept['Synonyms']:
#                 if syn:
#                     datapoints.append(
#                         {'input': syn,
#                          'output': output,
#                          'record_type': f'NCIT_{NCIT_TYPE}{record_type}:S0-F2',
#                          'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
#                          'snomed_id' : sid })
#             for definition in snomed_concept['Definitions']:
#                 if definition:
#                     datapoints.append(
#                         {'input': definition,
#                          'output': output,
#                          'record_type': f'NCIT_{NCIT_TYPE}{record_type}:S0-F2',
#                          'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
#                          'snomed_id' : sid })
#         return datapoints


#     def create_ncit_to_ncit():
#         datapoints = []
#         for syn in concept_detail[Synonyms]:
#             if syn:
#                 datapoints.append(
#                     {'input': syn,
#                      'output': output,
#                      'record_type': f'NCIT_{NCIT_TYPE}{record_type}:S2-F2',
#                      'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
#                      'snomed_id' : False })
        
#         # FOR FSN to TREE:
#         if output_id:
#             if FSN:
#                 datapoints.append(
#                     {'input': FSN,
#                      'output': output,
#                      'record_type': f'NCIT_{NCIT_TYPE}{record_type}:F2-F2',
#                      'id': f'NCIT_{NCIT_TYPE}-{ncitid}',
#                      'snomed_id' : False })
            
#         return datapoints

    
#     for ncitid , SNOMED_IDs in tqdm(dataset.items()):
#         SNOMED_IDs = flatten(SNOMED_IDs)
        
        
#         try:
#             concept_detail = ncit_concepts[ncitid] 
#         except:
#             non_existing_ncitids.append(ncitid)
#             continue

            
#         FSN = concept_detail[FullySpecifiedName][0]

#         if output_id:
#             output = ncit_to_treeid[ncitid]
#             record_type = "_tree"
#         else:
#             output = FSN
#             record_type = ''

        
#         # SNOMED to NCIT:
#         datapoins = create_snomed_datapoints_NCIT(SNOMED_IDs )
#         ncit_finetuning_eval = ncit_finetuning_eval + datapoins

#         # REST: NCIT TO NCIT:
#         datapoins = create_ncit_to_ncit()
#         ncit_finetuning_eval = ncit_finetuning_eval + datapoins


        
#     return ncit_finetuning_eval 

 

# %%
# ncit_pharm_finetuning_eval  = create_NCIT_pretraining_eval( eval_ids_NCIT_PHAR , NCIT_TYPE = "PHAR")
# ncit_neop_finetuning_eval  = create_NCIT_pretraining_eval( eval_ids_NCIT_NEOP , NCIT_TYPE = "NEOP")
# fma_finetuning_eval = create_FMA_pretraining_eval(eval_ids_FMA)

# # %%
# ncit_pharm_finetuning_eval_tree  = create_NCIT_pretraining_eval( eval_ids_NCIT_PHAR , NCIT_TYPE = "PHAR",output_id = True )
# ncit_neop_finetuning_eval_tree  = create_NCIT_pretraining_eval( eval_ids_NCIT_NEOP , NCIT_TYPE = "NEOP", output_id = True)
# fma_finetuning_eval_tree = create_FMA_pretraining_eval(eval_ids_FMA, output_id = True)

# %%
# all_finetuning_eval = ncit_pharm_finetuning_eval + ncit_neop_finetuning_eval + fma_finetuning_eval
# all_finetuning_eval_tree  = ncit_pharm_finetuning_eval_tree + ncit_neop_finetuning_eval_tree  + fma_finetuning_eval_tree 

# random.shuffle(all_finetuning_eval)
# random.shuffle(all_finetuning_eval_tree)

# # %%
# len(ncit_pharm_finetuning_eval_tree),  len(ncit_neop_finetuning_eval_tree ) , len( fma_finetuning_eval_tree )

# # %%
# len(all_finetuning_eval_tree)

# all_finetuining_eval_fsn_AND_tree = all_finetuning_eval + all_finetuning_eval_tree
# all_finetuining_train_fsn_AND_tree = all_finetuning_train + all_finetuning_train_tree

# random.shuffle(all_finetuining_eval_fsn_AND_tree)
# random.shuffle(all_finetuining_train_fsn_AND_tree)

# # %%
# len(all_finetuining_eval_fsn_AND_tree)

# # %%
# output_dir = f"{DATA_DIR}/finetuning/{today}_paperversion/"
# dumpjson(os.path.join(output_dir, f'all_finetuning_eval_tree.jsonl'), all_finetuning_eval_tree) 
# dumpjson(os.path.join(output_dir, f'all_finetuining_eval_fsn_AND_tree.jsonl'), all_finetuining_eval_fsn_AND_tree) 
# dumpjson(os.path.join(output_dir, f'all_finetuining_train_fsn_AND_tree.jsonl'), all_finetuining_train_fsn_AND_tree) 

######################### END: THIS BLOCK WOULD BE RESOLVED AFTER FIXES TO EVAL_DATA #########################


all_finetuning_train_tree_pharmbodyupsampled = ncit_pharm_finetuning_train_tree + ncit_neop_finetuning_train_tree + fma_finetuning_train_tree+fma_finetuning_train_tree  + ncit_pharm_finetuning_train_tree
all_finetuning_eval_tree_pharmbodyupsampled  = []#ncit_pharm_finetuning_eval_tree + ncit_neop_finetuning_eval_tree  + fma_finetuning_eval_tree +fma_finetuning_eval_tree  + ncit_pharm_finetuning_eval_tree

random.shuffle(all_finetuning_train_tree_pharmbodyupsampled)
random.shuffle(all_finetuning_eval_tree_pharmbodyupsampled)



list_of_exisiting_snomed_body_fsns = [i['input'] for i in fma_finetuning_train_tree]

# adding body structure data to fine-tune data
body_structure_concepts_fsn_to_treeid = []
for snomed_id in data_dict['snomed_body']:
    if "(body structure)" in snomed_concepts[snomed_id]['FullySpecifiedName'] and\
        snomed_concepts[snomed_id]['FullySpecifiedName'] not in list_of_exisiting_snomed_body_fsns:
        dic_tmp = {
         "input": snomed_concepts[snomed_id]['FullySpecifiedName'],
         "output": sid_to_treeid[snomed_id],
         "record_type": "SNOMED_FSN_tree:F0-F0",
         "id": f"SNOMED_BODY-{snomed_id}",
         "snomed_id": snomed_id}
        body_structure_concepts_fsn_to_treeid.append(dic_tmp)



output_dir = f"{DATA_DIR}/finetuning/{today}_pharmbody_upsampled/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dumpjson(os.path.join(output_dir, f'all_finetuning_train_tree_pharmbodyup_fixed.jsonl'), all_finetuning_train_tree_pharmbodyupsampled+body_structure_concepts_fsn_to_treeid) 
dumpjson(os.path.join(output_dir, f'all_finetuning_eval_tree_pharmbodyup_fixed.jsonl'), all_finetuning_eval_tree_pharmbodyupsampled) 

