

import os
import json
import zipfile
import random
import datetime
import csv
import io
import re
from collections import defaultdict
from pathlib import Path

from owlready2 import *

today = datetime.datetime.now().strftime("%m-%d-%y")
random.seed(42)

def def_type(def_id):
    def_name = None
    if def_id == '900000000000074008': #  Not sufficiently defined by necessary conditions definition status
        def_name =  'Primitive'
    elif def_id == '900000000000073002': # Sufficiently defined by necessary conditions definition status|
        def_name =  'Fully defined'
    else:
        raise ValueError(f"def id {def_id} is not expected.")
    return def_name


def read_descs(file_ref):
    fully_specified_name = defaultdict()
    synonyms = defaultdict(set)
    dictReader = csv.DictReader(io.TextIOWrapper(file_ref), delimiter="\t", quoting=csv.QUOTE_NONE)
    # iterate and set the dict with all terms
    for val in dictReader:
        if val['active'] == '1':  # MA: added to get active definitions only
            if val['typeId'] == '900000000000003001':
                fully_specified_name[val['conceptId']] = val['term']
            elif val['typeId'] == '900000000000013009':
                synonyms[val['conceptId']].add(val['term'])
    return fully_specified_name, synonyms


def read_concepts(file_ref):
    data_dict = defaultdict()
    dictReader = csv.DictReader(io.TextIOWrapper(file_ref), delimiter="\t", quoting=csv.QUOTE_NONE)
    # iterate and set the dict with all terms
    for val in dictReader:
        data_dict[val['id']] = {'active': val['active'] == '1', 'effective_time': val['effectiveTime'], 'def_type': def_type(val['definitionStatusId']) }
    return data_dict

def read_relations(file_ref):
    # https://confluence.ihtsdotools.org/display/DOCEXTPG/4.3.1.4+Relationships
    data_dict = defaultdict(set)
    dictReader = csv.DictReader(io.TextIOWrapper(file_ref), delimiter="\t", quoting=csv.QUOTE_NONE)
    # iterate and set the dict with all terms
    for i, v in enumerate(dictReader): 
        if v['active'] == '1':  #added to get active definitions only
            assert v['modifierId'] == '900000000000451002'
            data_dict[(v['sourceId'], v['destinationId'])].add(v['typeId'])
    return data_dict

def read_definitions(file_ref):
    data_dict = defaultdict(set)
    dictReader = csv.DictReader(io.TextIOWrapper(file_ref), delimiter="\t", quoting=csv.QUOTE_NONE)
    # iterate and set the dict with all terms
    for val in dictReader:
        if val['active'] == '1':
            assert val['typeId'] == '900000000000550004' # Definition (https://confluence.ihtsdotools.org/display/DOCGLOSS/description+type)
            data_dict[val['conceptId']].add(val['term'])
    return data_dict

def extract_text(fully_specified_term):
    return re.sub(r"\([^)]*\)", "", fully_specified_term).rstrip()

def extract_label(fully_specified_term):
    label = ''
    search_match = re.search(r"\([^)]*\)$", fully_specified_term.rstrip())
    if search_match is not None:
        label = re.sub(r"[^a-zA-Z0-9_\s]", "", search_match.group(0))
    return label

def extract_text_type(fully_specified_term):
    return extract_text(fully_specified_term), extract_label(fully_specified_term)



if __name__ == "__main__":

    ## output directory
    export_dir = '../data/ontology_json'
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    snomed_ontology_path_extension = '../data/source_data/SnomedCT_RF2Release_US1000124_20160901.zip' #from source




    extracted_dir = os.path.join(os.path.dirname(snomed_ontology_path_extension), 'extracted')
    Path(extracted_dir).mkdir(parents=True, exist_ok=True)

    print("generating SNOMED_CT ontology...")

    descs = None
    concepts = None
    relationships = None
    definitions = None

    with zipfile.ZipFile(snomed_ontology_path_extension) as zip_file:
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            if not filename: # skip directories
                continue
            if not (filename.endswith(".txt") and '/Snapshot/Terminology' in member and '__MACOSX' not in member): # use only the snapshot of current terminology, skip the rest
                continue
            with zip_file.open(member, 'r') as source:

                if 'Description_Snapshot' in member:
                    fully_specified_name, synonyms = read_descs(source)
                elif 'Concept_Snapshot' in member:
                    concepts = read_concepts(source)
                elif 'Relationship_Snapshot' in member and not 'StatedRelationship_Snapshot' in member:
                    relationships = read_relations(source)
                elif 'TextDefinition_Snapshot' in member:
                    definitions = read_definitions(source)




    children = defaultdict(list)
    parents = defaultdict(list)
    relationships_to = defaultdict(list)
    relationships_from = defaultdict(list)
    relationships_all = []
    for (f,t), rs in relationships.items():
        if f == '1240591000000102' or t == '1240591000000102':
            print(rs)
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


    import_dir = os.path.join(os.path.dirname(snomed_ontology_path_extension), 'to_import')
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
    print("saving SNOMED_CT ontology as:",os.path.join(export_dir,'snomed_ct_concepts20160901.json'))
    with open(os.path.join(export_dir,'snomed_ct_concepts20160901.json'), 'w',encoding='UTF-8') as f:
        json.dump(snomed_concepts, f)
