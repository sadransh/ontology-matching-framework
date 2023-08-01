 
# This Notebook generates three full ontologies (FMA, SNOMED CT, NCIT) and six subset ontologies (FMA (Body), SNOMED (Body), SNOMED (Neoplas), SNOMED (Pharm), , NCIT (Neoplas) and NCIT (Pharm)) in json format for UMLS task based on OAEI 2022::Bio-ML Track https://www.cs.ox.ac.uk/isg/projects/ConCur/oaei/2022/.
# 
# - For FMA full and all subset ontologies using owl files from the OAEI2022 data (.owl files)
# - For NCIT full and SNOMED full downlod the files from original source with the given OAEI version.
# - NCIT version used : V21.02d (.txt file) ,  SNOMED version used : US.2021.09.01 (.zip file), SNOMED version used : US.2016.09.01 (.zip file) as extention corpus
# 
# 
# - Download Snomed CT US Edition from https://www.nlm.nih.gov/healthit/snomedct/us_edition.html 
# 
# - Download NCIT from https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/2021/21.02d_Release/  (FLAT version)
# 

import os
import math
import json
import zipfile
import os
import logging
import random
import datetime
import re
today = datetime.datetime.now().strftime("%m-%d-%y")
random.seed(42)
logging.basicConfig(level=logging.DEBUG, format='%(process)d %(asctime)s %(levelname)s %(message)s')
logging.getLogger().setLevel(logging.DEBUG)
from pathlib import Path
from collections import defaultdict
import csv
import io
import csv
import os
from pathlib import Path
import pandas as pd
from owlready2 import *

export_dir = f'../data/ontology_json'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
## Input files : 

# raw ontologies --  downloaded from OAEI site or source

fma_ontology_path = '../data/source_data/UMLS/raw_data/fma_4.14.0.owl'  #using owl 
snomed_ontology_path = '../data/source_data/SnomedCT_USEditionRF2_PRODUCTION_20210901T120000Z.zip' #from source
snomed_ontology_path_extension = '../data/source_data/SnomedCT_RF2Release_US1000124_20160901.zip' #from source

ncit_ontology_path = f'../data/source_data/UMLS/raw_data/ncit_21.02d.owl' #using owl
ncit_ontology_path_for_hierarchy = f'../data/source_data/NCITThesaurus_v_21_02.txt' #using source 


# subset ontologies --  downloaded from OAEI site

subset_ontologies_path = '../data/source_data/UMLS/equiv_match/ontos'
fma_body_owl = "fma.body.owl"
ncit_neop_owl = "ncit.neoplas.owl"
ncit_phar_owl = "ncit.pharm.owl"
snomed_body_owl = "snomed.body.owl"
snomed_neop_owl = "snomed.neoplas.owl"
snomed_phar_owl = "snomed.pharm.owl"

# %%

def get_subset_ontology(full_concepts,subset_list):
    """Returns subset ontology for a given full ontology and codes associated 
    with subset ontology
    """
    d = {}
    for code in subset_list:
        d[code] = full_concepts[code]
    return d

# %%
def super_thing_classes_of(ent: ThingClass, ignore_root: bool = True):
    """Return super-classes of an entity class but excluding non-entity classes 
    such as existential axioms
    """
    supclasses = set()
    for supclass in ent.is_a:
        # ignore the root class Thing
        if isinstance(supclass, ThingClass):
            if ignore_root and supclass.name == "Thing":
                continue
            else:
                supclasses.add(supclass)
    return list(supclasses)




print("generating full FMA ontology ...")

fma_full = get_ontology(fma_ontology_path).load()

fma_full.base_iri = 'fma_full'
fma_full_entities = list(fma_full.classes())

fma_concepts_all = {}
for cl in fma_full.classes():
    d = {"FMAID" : cl.FMAID,
         "Preferred_name" :cl.preferred_name,
         "Label" : cl.label,
         "English" : cl.English_equivalent,
         "ABBR" : cl.abbreviation,
         "Definition" : cl.definition,
         "Name" :cl.name,
         "Other_equivalents" : cl.other_Latin_equivalents,
         "Slot_syn" : cl.slot_synonym,
         "Synonym" :cl.synonym,
         "Status" : cl.term_status,
         "Ancestors": [i.name for i in cl.ancestors()],
         "Parent" : [i.name for i in super_thing_classes_of(cl)]}
    fma_concepts_all[cl.name] = d

print("generated FMA with entities length of ",len(fma_full_entities), "and concepts count of: " , len(fma_concepts_all))


print("saving FMA as:",os.path.join(export_dir,"fma_concepts_all.json"))

with open(os.path.join(export_dir,"fma_concepts_all.json"), 'w') as f:
    json.dump(fma_concepts_all,f)

with open(os.path.join(export_dir,"fma_concepts_all.json"), 'r') as f:
    fma_concepts = json.load(f)
    

print("generating FMA subset...")


fma_bodyonto = get_ontology(os.path.join(subset_ontologies_path, fma_body_owl)).load()
fma_bodyonto.base_iri = 'fma_body'
fma_body_entities = list(fma_bodyonto.classes())
fma_body_entities_name = [i.name for i in fma_bodyonto.classes()]
len(fma_body_entities_name),fma_body_entities_name[:5]


fma_body_concepts = get_subset_ontology(fma_concepts,fma_body_entities_name)

print("saving FMA subset as:",os.path.join(export_dir,"fma_concepts_body.json"))

with open(os.path.join(export_dir,"fma_concepts_body.json"), 'w') as f:
    json.dump(fma_body_concepts,f)




print("generating SNOMED_CT ontology...")



extracted_dir = os.path.join(os.path.dirname(snomed_ontology_path), 'extracted')
Path(extracted_dir).mkdir(parents=True, exist_ok=True)

def desc_type(type_id):
    type_name = None
    if type_id == '900000000000003001':
        type_name =  'Fully specified name'
    elif type_id == '900000000000013009':
        type_name =  'Synonym'
    elif type_id == '900000000000550004':
        type_name =  'Definition'
    else:
        raise ValueError(f"type id {type_id} is not expected. See https://confluence.ihtsdotools.org/display/DOCGLOSS/description+type")
    return type_name

def def_type(def_id):
    def_name = None
    if def_id == '900000000000074008': #  Not sufficiently defined by necessary conditions definition status
        def_name =  'Primitive'
    elif def_id == '900000000000073002': # Sufficiently defined by necessary conditions definition status|
        def_name =  'Fully defined'
    else:
        raise ValueError(f"def id {def_id} is not expected.")
    return def_name    
            
def read_descs(f):
    fully_specified_name = defaultdict()
    synonyms = defaultdict(set)
    dictReader = csv.DictReader(io.TextIOWrapper(f), delimiter="\t", quoting=csv.QUOTE_NONE)

    for i, v in enumerate(dictReader):
        if v['active'] == '1':  
            if v['typeId'] == '900000000000003001':
                fully_specified_name[v['conceptId']] = v['term']
            elif v['typeId'] == '900000000000013009':
                synonyms[v['conceptId']].add(v['term'])
    return fully_specified_name, synonyms


def read_concepts(f):
    data_dict = defaultdict()
    dictReader = csv.DictReader(io.TextIOWrapper(f), delimiter="\t", quoting=csv.QUOTE_NONE)

    for i, v in enumerate(dictReader):
        data_dict[v['id']] = {'active': v['active'] == '1', 'effective_time': v['effectiveTime'], 'def_type': def_type(v['definitionStatusId']) }
    return data_dict

def read_relations(f): # https://confluence.ihtsdotools.org/display/DOCEXTPG/4.3.1.4+Relationships
    data_dict = defaultdict(set)
    dictReader = csv.DictReader(io.TextIOWrapper(f), delimiter="\t", quoting=csv.QUOTE_NONE)
    # iterate and set the dict with all terms
    for i, v in enumerate(dictReader): 
        if v['active'] == '1':  # MA: added to get active definitions only
            assert v['modifierId'] == '900000000000451002'
            data_dict[(v['sourceId'], v['destinationId'])].add(v['typeId'])
    return data_dict

def read_definitions(f):
    data_dict = defaultdict(set)
    dictReader = csv.DictReader(io.TextIOWrapper(f), delimiter="\t", quoting=csv.QUOTE_NONE)
    # iterate and set the dict with all terms
    for i, v in enumerate(dictReader):
        if v['active'] == '1':
            assert v['typeId'] == '900000000000550004' # Definition (https://confluence.ihtsdotools.org/display/DOCGLOSS/description+type)
            data_dict[v['conceptId']].add(v['term'])
    return data_dict


descs = None
concepts = None
relationships = None
definitions = None


with zipfile.ZipFile(snomed_ontology_path) as zip_file:
    
    for member in zip_file.namelist():
        filename = os.path.basename(member)
        
        if not filename: # skip directories
            continue
        if not (filename.endswith(".txt") and '/Snapshot/Terminology' in member and '__MACOSX' not in member): # use only the snapshot of current terminology, skip the rest
            continue
        
        with zip_file.open(member, 'r') as source:
            #print(member)
            if 'Description_Snapshot' in member:
                fully_specified_name, synonyms = read_descs(source)
            elif 'Concept_Snapshot' in member:
                concepts = read_concepts(source)
            elif 'Relationship_Snapshot' in member and not 'StatedRelationship_Snapshot' in member:
                relationships = read_relations(source)
            elif 'TextDefinition_Snapshot' in member:
                definitions = read_definitions(source)


def extract_text(fully_specified_name):
    return re.sub(r"\([^)]*\)", "", fully_specified_name).rstrip()

def extract_label(fully_specified_name):
    label = ''
    search_match = re.search(r"\([^)]*\)$", fully_specified_name.rstrip())
    if search_match is not None:
        label = re.sub(r"[^a-zA-Z0-9_\s]", "", search_match.group(0))
    return label

children = defaultdict(list)
parents = defaultdict(list)
relationships_to = defaultdict(list)
relationships_from = defaultdict(list)
relationships_all = []
for (f,t), rs in relationships.items():

    for r in rs:
        fsn = extract_text(fully_specified_name[r])
        if fsn == 'Is a':
            children[t].append(f)
            parents[f].append(t)
        else:
            relationships_to[t].append((f, fsn))
            relationships_from[f].append((t, fsn))
        relationships_all.append((f,t, fsn))

concepts_all = []
for c_id, c in concepts.items():
    fsn_txt = extract_text(fully_specified_name[c_id])
    concept_type = extract_label(fully_specified_name[c_id])
    definitions[c_id].discard(fsn_txt)
    synonyms[c_id].discard(fsn_txt)
    concepts_all.append((c_id, c['effective_time'], fsn_txt, '|'.join(synonyms[c_id]), '|'.join(definitions[c_id]), f"{c['def_type'].upper()}|{'ACTIVE' if c['active'] else 'INACTIVE'}|{concept_type.upper() if concept_type else 'NO_TYPE'}"))


import_dir = os.path.join(os.path.dirname(snomed_ontology_path), 'to_import')
Path(import_dir).mkdir(parents=True, exist_ok=True)

fieldnames=['ConceptId:ID', 'EffectiveDate', 'FullySpecifiedName', 'Synonyms', 'Definitions',':LABEL']
with open(os.path.join(import_dir, 'concepts.csv'), 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for concept in concepts_all:
        writer.writerow(concept)

fieldnames=[':START_ID', ':END_ID', ':TYPE']
with open(os.path.join(import_dir, 'relationship.csv'), 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for relationship in relationships_all:
        writer.writerow((relationship[0], relationship[1], relationship[2].upper().replace(' ', '_')))


snomed_concepts = {}
def extract_text_type(fully_specified_name):
    return extract_text(fully_specified_name), extract_label(fully_specified_name)
for c_id, c in concepts.items():
    if not c['active']:
        continue
    fsn_txt, _ = extract_text_type(fully_specified_name[c_id])
    definitions[c_id].discard(fsn_txt)
    synonyms[c_id].discard(fsn_txt)
    ps = sorted(parents[c_id], key=lambda x: fully_specified_name[x])
    ps_txt = [fully_specified_name[pi] for pi in ps]
    cs = sorted(children[c_id], key=lambda x: fully_specified_name[x])
    cs_txt = [fully_specified_name[ci] for ci in cs]
    relations = sorted([(rt, extract_text(fully_specified_name[f])) for f, rt in relationships_from[c_id]], key=lambda x: x[0])
    snomed_concepts[c_id] = {'Relations': relations, 'ConceptId': c_id, 'ParentsIds': ps, 'ChildrenIds': cs, 'Parents': ps_txt, 'Children': cs_txt, 'FullySpecifiedName': fully_specified_name[c_id], 'Definitions': sorted(list(definitions[c_id])), 'Synonyms':  sorted(list(synonyms[c_id])),  'DefinitionType': c['def_type']}

#saving
print("saving SNOMED_CT ontology as:",os.path.join(export_dir,'snomed_concepts_all.json'))
with open(os.path.join(export_dir,'snomed_concepts_all.json'), 'w') as f:
    json.dump(snomed_concepts, f)


with open(os.path.join(export_dir,'snomed_concepts_all.json'), 'r') as f:
    snomed_concepts = json.load(f)

print("generating SNOMED_CT subset ontologies ...")

snomed_pharonto = get_ontology(os.path.join(subset_ontologies_path, snomed_phar_owl)).load()
snomed_pharonto.base_iri = "snomed_phar"
snomed_phar_entities = list(snomed_pharonto.classes())
snomed_phar_entities_name = [i.name for i in snomed_pharonto.classes()]
len(snomed_phar_entities_name),snomed_phar_entities_name[:5]


snomed_bodyonto = get_ontology(os.path.join(subset_ontologies_path, snomed_body_owl)).load()
snomed_bodyonto.base_iri = "snomed_body"
snomed_body_entities = list(snomed_bodyonto.classes())
snomed_body_entities_name = [i.name for i in snomed_bodyonto.classes()]
len(snomed_body_entities_name),snomed_body_entities_name[:5]


snomed_neoponto = get_ontology(os.path.join(subset_ontologies_path, snomed_neop_owl)).load()
snomed_neoponto.base_iri = "snomed_neop"
snomed_neop_entities = list(snomed_neoponto.classes())
snomed_neop_entities_name = [i.name for i in snomed_neoponto.classes()]
len(snomed_neop_entities_name),snomed_neop_entities_name[:5]

snomed_phar_concepts = get_subset_ontology(snomed_concepts,snomed_phar_entities_name)
snomed_body_concepts = get_subset_ontology(snomed_concepts,snomed_body_entities_name)
snomed_neop_concepts = get_subset_ontology(snomed_concepts,snomed_neop_entities_name)


print("saving SNOMED_CT subset phar as",os.path.join(export_dir,"snomed_phar_concepts.json"))

with open(os.path.join(export_dir,"snomed_phar_concepts.json"), 'w') as f:
    json.dump(snomed_phar_concepts,f)
print("saving SNOMED_CT subset body as",os.path.join(export_dir,"snomed_body_concepts.json"))

with open(os.path.join(export_dir,"snomed_body_concepts.json"), 'w') as f:
    json.dump(snomed_body_concepts,f)
print("saving SNOMED_CT subset neop as",os.path.join(export_dir,"snomed_neop_concepts.json"))

with open(os.path.join(export_dir,"snomed_neop_concepts.json"), 'w') as f:
    json.dump(snomed_neop_concepts,f)


print("generating NCIT ...")

## get the labels from OWL files
ncit_full = get_ontology(ncit_ontology_path).load()
ncit_full_entities = list(ncit_full.classes())
ncit_full_entities_name = [i.name for i in ncit_full.classes()]



# some of the parents were missing in the owl file, so using the source file to extract hierarchy.
df_OM22 = pd.read_csv(ncit_ontology_path_for_hierarchy, dtype = str,delimiter = '\t',header=None).fillna('')
df_OM22.columns = ['Code','IRI','Parents','Synonyms','Definition',\
                   'DisplayName','Status', 'SemanticType']
df_OM22 = df_OM22[['Code','Parents','Synonyms','Definition','DisplayName','Status', 'SemanticType']]
all_codes_OM22 = df_OM22.Code.tolist()
ncit_concepts_source = df_OM22.set_index('Code').to_dict(orient='index')

# genrating dictionary by combine owl file and source file attributes
ncit_concepts_all = {}
for cl in ncit_full.classes():
    d = {"Code" : cl.name,
         "Preferred_name" :cl.P108,
         "Label" : cl.label,
         "Definition" : cl.P97,
         "Name" :cl.name,
         "Synonym" : list(set([i for i in cl.P90 if i not in cl.label])),
         "legacy_concept_name" : cl.P366,
         "Parent" : ncit_concepts_source[cl.name]['Parents'].split('|')}
    ncit_concepts_all[cl.name] = d

with open(os.path.join(export_dir,"ncit_concepts_all.json"), 'w') as f:
    json.dump(ncit_concepts_all,f)

print("generating NCIT subsets ...")


with open(os.path.join(export_dir,"ncit_concepts_all.json"), 'r') as f:
    ncit_concepts = json.load(f)
    

ncit_neoponto = get_ontology(os.path.join(subset_ontologies_path, ncit_neop_owl)).load()
ncit_neop_entities = list(ncit_neoponto.classes())
ncit_neop_entities_name = [i.name for i in ncit_neoponto.classes()]
len(ncit_neop_entities_name),ncit_neop_entities_name[:5]


ncit_pharonto = get_ontology(os.path.join(subset_ontologies_path, ncit_phar_owl)).load()
ncit_phar_entities = list(ncit_pharonto.classes())
ncit_phar_entities_name = [i.name for i in ncit_pharonto.classes()]
len(ncit_phar_entities_name),ncit_phar_entities_name[:5]

ncit_phar_concepts = get_subset_ontology(ncit_concepts,ncit_phar_entities_name)
ncit_neop_concepts = get_subset_ontology(ncit_concepts,ncit_neop_entities_name)

print("saving NCIT subsets ...")

with open(os.path.join(export_dir,"ncit_phar_concepts.json"), 'w') as f:
    json.dump(ncit_phar_concepts,f)
with open(os.path.join(export_dir,"ncit_neop_concepts.json"), 'w') as f:
    json.dump(ncit_neop_concepts,f)
