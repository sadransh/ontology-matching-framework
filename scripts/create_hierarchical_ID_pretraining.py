""" This script creates the pre-training data for the hierarchical ID's."""

import json
from collections import defaultdict
import re
import os
import datetime
import random
from pathlib import Path
today = datetime.datetime.now().strftime("%m-%d-%y")


def dumpjson(output_directory, data):
    """Dumps data to a jsonl file."""
    output_directory = Path(output_directory)
    output_directory.parent.mkdir(exist_ok=True, parents=True)
    with open(output_directory, 'w', encoding='utf-8') as file_pointer:
        for d in data:
            file_pointer.write(json.dumps(d))
            file_pointer.write('\n')

#loading data
print("loading ontologies ....")
DATA_DIR = '../data'
with open(f"{DATA_DIR}/ontology_json/snomed_concepts_all.json", "r", encoding='UTF-8') as fp:
    snomed_concepts = json.load(fp)

with open(f"{DATA_DIR}/ontology_json/fma_concepts_all.json", "r", encoding='UTF-8') as fp:
    fma_concepts = json.load(fp)

with open(f"{DATA_DIR}/ontology_json/ncit_concepts_all.json", "r", encoding='UTF-8') as fp:
    ncit_concepts = json.load(fp)


# ncit_concepts corrections for consistency with other ontologies
for k,v in ncit_concepts.items():
    if 'Label' not in v or len(v['Label'])==0:
        ncit_concepts[k]['Label'] = v['Preferred_name']
    if v["Parent"]  == []:
        ncit_concepts[k]["Parents"] = [""]
    else:
        ncit_concepts[k]["Parents"] = v["Parent"]


# ### Create Children ID for FMA and NCIT:
# Create Children IDS:
def add_children_ids(concept_details):
    """Adds children ids to the concept details dictionary.

    Args:
        concept_details (Dict of Dict): dictionary of concepts details

    Returns:
        Dict: dictionary of concepts details with children ids added
    """
    root_childrens = []

    for concept_id, concept_detail in concept_details.items():
        concept_details[concept_id] = defaultdict(list, concept_detail)
    string_match = re.match("\D+",concept_id)[0]

    for concept_id, concept_detail in concept_details.items():
        for parent in concept_detail["Parents"]:
            if concept_details.get(parent):
                if concept_id not in concept_details[parent]["ChildrenIds"]:
                    concept_details[parent]["ChildrenIds"].append(concept_id)
            else:
                pass
                # print(f"The {FID} has a parent called: {parent}, which does not exist in our Concept Dict")

                if concept_id not in root_childrens:
                    root_childrens.append(concept_id)

                    concept_details[concept_id]["Parents"].remove(parent)
                    concept_details[concept_id]["Parents"].append(f"{string_match}root")

    concept_details[f"{string_match}root"] = dict()
    for key, value in concept_detail.items():
        if isinstance(value,str):
            concept_details[f"{string_match}root"][key] = ""
        elif  isinstance(value,list):
            concept_details[f"{string_match}root"][key] = [""]
        else:
            concept_details[f"{string_match}root"][key] = False

    concept_details[f"{string_match}root"]["definition"] = "Root of the concept Tree"
    concept_details[f"{string_match}root"]["ChildrenIds"] = root_childrens
    return concept_details

del fma_concepts['Agent']
for key, value in fma_concepts.items():
    if value["Parent"]  == []:
        fma_concepts[key]["Parents"] = [""]
    else:
        fma_concepts[key]["Parents"] = value["Parent"] 

fma_concepts = add_children_ids(fma_concepts)
ncit_concepts = add_children_ids(ncit_concepts)

### Build the ids for SNOMED, FMA and NCIT:

class TreeNode:
    """
    A node in a tree. Each node has an ID, a name, and a list of children.
    The tree is used to build the SNOMEDCT structure.
    """

    def __init__(self, data):
        """ Creates a node in a tree."""
        self.data = data
        self.children = []
        self.parent = None
        self.new_id = None
        self.syn_ids = []

    def add_child(self, child):
        """ Adds a child to the node."""
        child.parent = self
        self.children.append(child)

    def get_level(self):
        """ Returns the level of the node."""
        level = 0
        node = self
        while node.parent:
            node = node.parent
            level += 1
        return level

    def print_tree(self , new_id = False ):
        """ Prints the tree."""
        spaces = " " * self.get_level() * 3
        prefix = spaces + "|--" if self.parent else ""
        if new_id:
            print(prefix + self.new_id)
        else:
            print(prefix + self.data)
        if self.children:
            for child in self.children:
                child.print_tree(new_id)

    def print_parent(self , new_id = False):
        spaces = " " * self.get_level() * 3
        prefix = spaces + "|--" if self.parent else ""
        if new_id:
            print(prefix + self.new_id)
        else:  
            print(prefix + self.data)
        if self.parent:
            self.parent.print_parent(new_id) 


    def populate_ids(self, parentID, child_counter):
        populated_id = parentID+"-"+child_counter
        if not self.new_id:
            if child_counter:
                self.new_id = populated_id
            else:
                self.new_id = parentID
        else: 
            self.syn_ids.append(populated_id)
        if child_counter:
            parentID = populated_id

        if self.children:

            for child_counter,child in enumerate(self.children):

                child.populate_ids( parentID, str(child_counter))


# %%
def build_snomed_tree(snomed_id , concept_details):
    """Builds a tree of SNOMEDCT concepts.
    snomed_id = "138875005" is the SNOMED ROOT


    Args:
        snomed_id (STRING): snomed_id of the root node of the tree
        concept_details (_type_): dictionary of concepts from extract_concept_details() function

    Returns:
        TreeNode: root node of the tree
        hierarchical_id_dict (dict): dictionary of tree nodes, with the key being the snomed_id and the value being a list of corresponding tree nodes.
    """

    def build_tree_loop(parent, parent_id, hierarchical_id_dict):

        for children_id in concept_details[parent_id]["ChildrenIds"]:
            child = TreeNode(children_id)
            parent.add_child(child)

            hierarchical_id_dict[children_id].append(child)

            build_tree_loop(child, children_id, hierarchical_id_dict)
        return

    root = TreeNode(snomed_id)
    hierarchical_id_dict = defaultdict(list)
    hierarchical_id_dict[snomed_id].append(root)

    for children_id in concept_details[snomed_id]["ChildrenIds"]:
        child = TreeNode(children_id)
        root.add_child(child)
        hierarchical_id_dict[children_id].append(child)
        build_tree_loop(child, children_id, hierarchical_id_dict)
    
        
    return root, hierarchical_id_dict

def create_mapping_hierarchicalId_conceptid(hierarchical_id_dict):
    """Finds the shortest hierarchical-id and all the synonyms for each ID
    Also create the hierarchical_ID to the ID

    Args:
        hierarchical_id_dict (dict of hierarchicals): dictionary of hierarchical ids created by build_snomed_tree()

    Returns:
        hierarchical0_id_dict: dictionary of hierarchical_id to hierarchical structure with all the synonyms
        id_to_hierarchicalId: dictionary of concept id to hierarchical_id
    """
    has_multiple_names = False
    count_multiple = 0
    hierarchical0_id_dict = dict()
    id_to_hierarchicalId = dict()

    for c_id, trees in hierarchical_id_dict.items():
        h0 = trees[0]
        h0.syn_ids = [hid.new_id for hid in hierarchical_id_dict[c_id]]
        h0.min_id = min(h0.syn_ids, key=len)
        hierarchical0_id_dict[c_id] = h0
        id_to_hierarchicalId[c_id] = h0.min_id

        if len(trees) > 1:
            has_multiple_names = True
            count_multiple += 1

    if has_multiple_names:
        print(f" {round((100*count_multiple)/len(hierarchical_id_dict),2)}% of this ontology have multiple hierarchical-ID per each concept")
    else:
        print("This ontology does not have multiple hierarchical IDS.")

    return hierarchical0_id_dict, id_to_hierarchicalId


#Build and populate the SNOMED tree:

SNOMED_ROOT = "138875005"
snomed_root, snomed_hierarchical_id_dict  = build_snomed_tree(SNOMED_ROOT , snomed_concepts)
snomed_root.populate_ids("0","")

#Build and populate the FMA tree:
fma_root, fma_hierarchical_id_dict  = build_snomed_tree("fmaroot" , fma_concepts)
fma_root.populate_ids("1","")

#Build and populate the NCIT tree:
ncit_root, ncit_hierarchical_id_dict  = build_snomed_tree("Croot" , ncit_concepts)
ncit_root.populate_ids("2","")

snomed_hierarchical0_id_dict , snomed_to_hierarchicalId = create_mapping_hierarchicalId_conceptid(snomed_hierarchical_id_dict)
fma_hierarchical0_id_dict , fma_to_hierarchicalId = create_mapping_hierarchicalId_conceptid(fma_hierarchical_id_dict)
ncit_hierarchical0_id_dict , ncit_to_hierarchicalId = create_mapping_hierarchicalId_conceptid(ncit_hierarchical_id_dict)



def extract_text(fully_specified_name):
    """This function extracts the SMOMED name given a SNOMED CT definitions.
    Basically, as all the definitions have their type infront of them, we can
    extract the name by removing the type.
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


# Save the hiraricalIDs as json:

output_dir = f"{DATA_DIR}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

with open(os.path.join(output_dir  ,"omversion_snomed_to_hierarchicalId.json"), "w", encoding='UTF-8') as fp:
    json.dump(snomed_to_hierarchicalId, fp)

with open(os.path.join(output_dir  ,"omversion_ncit_to_hierarchicalId.json"), "w", encoding='UTF-8') as fp:
    json.dump(ncit_to_hierarchicalId, fp)

with open(os.path.join(output_dir  ,"omversion_fma_to_hierarchicalId.json"), "w", encoding='UTF-8') as fp:
    json.dump(fma_to_hierarchicalId, fp)

print(os.path.join(output_dir  ,"omversion_fma_to_hierarchicalId.json"))



# ### Lets extract pre-training data from the ontologies:

print("saving fhierarchical Id per ontology ")
with open(f"{DATA_DIR}/omversion_fma_to_hierarchicalId.json", "r", encoding='UTF-8') as fp:
    fma_to_hierarchicalId = json.load(fp)
hierarchicalId_to_fma = { v:k for k,v in fma_to_hierarchicalId.items()}

# Load hierarchicalId:
with open(f"{DATA_DIR}/omversion_ncit_to_hierarchicalId.json", "r", encoding='UTF-8') as fp:
    ncit_to_hierarchicalId = json.load(fp)
hierarchicalId_to_ncit = { v:k for k,v in ncit_to_hierarchicalId.items()}

# Load SNOMED hierarchicalId:
with open(f"{DATA_DIR}/omversion_snomed_to_hierarchicalId.json", "r", encoding='UTF-8') as fp:
    snomed_to_hierarchicalId = json.load(fp)
hierarchicalId_id_to_snomed = { v:k for k,v in snomed_to_hierarchicalId.items()}


# %%
def create_pre_training_data_gt(concept_details, sid_to_nid, record_type= "SNOMED_CT_GRAPH" , give_list=False, extract_multiples_only=False):
    """Creates a parent to child relation based on the new defined IDs.
    For pre-training: the output of this will go thorough Masking processes before each training iteration
    For fine-tuning: we will use the output to train a model that given a child can detect the parents.

    Args:
        parents (dict): dictionary of parents
        sid_to_nid (dict): dictionary of snomed_id to new ID
        give_list (bool, optional): if True, will return a list, othervise will return a dict
        extract_multiples_only (bool, optional): if True, will extract only the ones with multiple parents.

    Returns:
        list of dict: list of samples
    """

    train_samples = []
    train_samples_list = []
    counter = 0
    def_type = record_type.split("_")[0]

    for child_id, concept_detail in concept_details.items():

        if "root" in child_id :
            continue
        try:
            if def_type == "SNOMED":
                parent_ids = concept_detail['ParentsIds']
            else:
                parent_ids = concept_detail['Parents']
        except:
            print(concept_detail)

        c_nid = sid_to_nid[child_id]
        if extract_multiples_only:
            # If there are multiple parents PASS
            if len(parent_ids) < 2:
                continue
        multiple_parents = [sid_to_nid[sid] for sid in  parent_ids]
        
        if give_list:
            train_samples_list.append([c_nid, multiple_parents])
        train_samples.append(
            {
                "data": [c_nid, sorted(multiple_parents)],
                "record_type": record_type,
                "id": f"{record_type}-{str(counter)}",
                "concept_id" : f"{def_type}-{child_id}",
            }
        )
        counter += 1
    if extract_multiples_only:
        print(f"{round((counter*100)/len(concept_details),2)}% of {def_type} ontology have multiple hierarchical-ID per each concept")
    if give_list:
        return train_samples_list
    return train_samples


pre_training_NCIT_CtoP = create_pre_training_data_gt(ncit_concepts, ncit_to_hierarchicalId, record_type= "NCIT_TREE_CP", give_list=False, extract_multiples_only=True )
pre_training_FMA_CtoP = create_pre_training_data_gt(fma_concepts, fma_to_hierarchicalId, record_type= "FMA_TREE_CP", give_list=False, extract_multiples_only=True )
pre_training_SNOMED_CtoP = create_pre_training_data_gt(snomed_concepts, snomed_to_hierarchicalId, record_type= "SNOMED_TREE_CP", give_list=False, extract_multiples_only=True )



#create a pre-training data for hierarchical ids that trains only on the synonym ID's dataset:

def create_pre_training_data_gt_for_tree_syn(hierarchical0_id_dict , sid_to_hierarchicalId, record_type= "SNOMED_CT_SYN_TREE"):
    """Creates Synonym relations based on the new defined Tree IDs.
    
    Args:
        hierarchical0_id_dict (dict): dictionary of hierarchical[0]s
        sid_to_hierarchicalId (dict): dictionary of snomed_id to new ID
        record_type (string, optional): record type so we can use when preprocessing on training side. Defaults to SNOMED_CT_SYN_TREE.

    Returns:
        list of dict: list of samples
    """
    
    train_samples = []
    counter = 0
    def_type = record_type.split("_")[0]
    
    for sid, tree in hierarchical0_id_dict.items():
        synonyms  = hierarchical0_id_dict[sid].syn_ids
        
        if len(synonyms)>1:
            train_samples.append(
                {
                    "data": sorted(synonyms),
                    "record_type": record_type,
                    "id": f"{record_type}-{str(counter)}",
                    "concept_id" : f"{def_type}-{sid}",
                }
            )
            counter += 1
    print(f"There are {counter} samples for the {record_type} record.") 
    return train_samples


pre_training_NCIT_hierarchicalId_syns = create_pre_training_data_gt_for_tree_syn(ncit_hierarchical0_id_dict, ncit_to_hierarchicalId , "NCIT_SYN_TREE")
pre_training_SNOMED_hierarchicalId_syns = create_pre_training_data_gt_for_tree_syn(snomed_hierarchical0_id_dict, snomed_to_hierarchicalId , "SNOMED_SYN_TREE")

# #### Create Tree-ID to ENG (FSN, SYN, DEF):

def create_pre_training_data_id_to_english(concept_details, sid_to_nid, record_type, ontology_name, USE_DEF = True):
    """Creates pre-training data for the id to concept name (+Syn) mapping.

    Args:
        concept_details (dict): dictionary of concepts from extract_concept_details() function
        sid_to_nid (dict): dictionary of snomed_id to new ID
        record_type (string): record type so we can use when preprocessing on training side.

    Returns:
        list of dict: list of samples
    """

    train_samples = []
    counter_fsn = 0
    counter_def = 0
    counter_syn = 0
    
    if ontology_name == "SNOMED":
        FullySpecifiedName = "FullySpecifiedName"
        Synonyms = "Synonyms"
        Definitions = "Definitions"
    elif ontology_name == "FMA":
        Synonyms = "Synonym"
        FullySpecifiedName = "Label"
        Definitions = "Definition"
    elif ontology_name == "NCIT":
        Synonyms = "Synonym"
        FullySpecifiedName = "Label"
        Definitions = "Definition"
    else:
        print("invalid ontology name")
        return()
    
   
    for sid, concept_detail in concept_details.items():

        hierarchicalId = sid_to_nid[sid]
        fully_specifiedName = extract_text(concept_detail[FullySpecifiedName]) if ontology_name == "SNOMED" else concept_detail[FullySpecifiedName][0]
            
        if fully_specifiedName:
            train_samples.append(
                {
                    "data": [hierarchicalId, fully_specifiedName],
                    "record_type": f"{record_type}_TREE_TO_FSN",
                    "id": f"{record_type}_TREE_TO_FSN-{str(counter_fsn)}",
                    "concept_id" : f"{ontology_name}-{sid}",
                }
            )

            counter_fsn += 1

        for syn in concept_detail[Synonyms]:
            if syn:
                train_samples.append(
                    {
                        "data": [hierarchicalId, syn],
                        "record_type":  f"{record_type}_TREE_TO_SYN",
                        "id": f"{record_type}_TREE_TO_SYN-{str(counter_syn)}",
                        "concept_id" : f"{ontology_name}-{sid}",
                    }
                )
                counter_syn += 1
    
        if USE_DEF:
            for definition in concept_detail[Definitions]:
                if definition:
                    train_samples.append(
                        {
                            "data": [hierarchicalId, definition],
                            "record_type": f"{record_type}_TREE_TO_DEF",
                            "id": f"{record_type}_TREE_TO_DEF-{str(counter_def)}",
                            "concept_id" : f"{ontology_name}-{sid}",
                        }
                    )
                    counter_def += 1

    print(f"  *** For {ontology_name}, out of {len(train_samples)} samples:")
    print(f"{round((100*counter_fsn)/len(train_samples),2)}% data are TREE_TO_FSN ({counter_fsn}),")
    print(f"{round((100*counter_syn)/len(train_samples),2)}% data are TREE_to_SYN({counter_syn}),")
    print(f"{round((100*counter_def)/len(train_samples),2)}% data are TREE_to_DEF ({counter_def}).\n")
            
    return train_samples

# %%
pre_training_FMA_tree_to_name = create_pre_training_data_id_to_english(fma_concepts , fma_to_hierarchicalId , "FMA", "FMA", USE_DEF=False)
pre_training_NCIT_tree_to_name = create_pre_training_data_id_to_english(ncit_concepts , ncit_to_hierarchicalId , "NCIT", "NCIT" , USE_DEF=False)
pre_training_SNOMED_tree_to_name = create_pre_training_data_id_to_english(snomed_concepts , snomed_to_hierarchicalId , "SNOMED" ,"SNOMED" ,USE_DEF= True)


# #### Create a FullySpecifiedName to Syn, Def to Syn, Def to FSN task:

def fspn_to_syns(concept_details, record_type="SNOMED", ontology_name="SNOMED" , USE_DEF=True ,def_to_syn=True):
    """Creates pre-training data for the FSPN to Synonyms (+Def) mapping.

    Args:
        concept_details (dict): dictionary of concepts from extract_concept_details() function
        record_type (string, optional): record type so we can use when preprocessing on training side. Defaults to "SNOMED_CT_FSPN_to".
        ontology_name (string, optional): name of the ontology. Defaults to "SNOMED"
        def_to_syn (bool, optional): If you want to also add the def_to_syn to the data. Defaults to False.

    Returns:
        list of dict: list of samples
    """
      
    if ontology_name == "SNOMED":
        FullySpecifiedName = "FullySpecifiedName"
        Synonyms = "Synonyms"
        Definitions = "Definitions"
    elif ontology_name == "FMA":
        # FullySpecifiedName = "PreferredLabel"
        Synonyms = "Synonym"
        FullySpecifiedName = "Label"
        Definitions = "Definition"
    elif ontology_name == "NCIT":
        Synonyms = "Synonym"
        # FullySpecifiedName = "DisplayName"
        FullySpecifiedName = "Label"
        Definitions = "Definition"
    else:
        print("invalid ontology name")
        return()
    
    
    train_samples = list()
    syn_counter = 0
    def_counter = 0
    def_to_syn_counter = 0

    for sid , concept_detail in concept_details.items():

        fully_specifiedName = extract_text(concept_detail[FullySpecifiedName]) if ontology_name == "SNOMED" else concept_detail[FullySpecifiedName][0]

        for syn in concept_detail[Synonyms]:
            if syn:
                train_samples.append(
                    {
                        "data": [fully_specifiedName, syn],
                        "record_type": f"{record_type}_FSN_to_SYN",
                        "id": f"{record_type}_FSN_to_SYN-{str(syn_counter)}",
                        "concept_id" : f"{ontology_name}-{sid}",
                    }
                )
                syn_counter += 1
                
        if USE_DEF:
            for definition in concept_detail[Definitions]:
                if definition:
                    train_samples.append(
                        {
                            "data": [fully_specifiedName, definition],
                            "record_type": f"{record_type}_FSN_to_DEF",
                            "id": f"{record_type}_FSN_to_DEF-{str(def_counter)}",
                            "concept_id" : f"{ontology_name}-{sid}",
                        }
                    )
                    def_counter += 1

                    if def_to_syn:
                        for syn in concept_detail["Synonyms"]:
                            if syn:
                                train_samples.append(
                                    {
                                        "data": [definition, syn],
                                        "record_type": f"{record_type}_DEF_to_SYN",
                                        "id": f"{record_type}_DEF_to_SYN-{str(def_to_syn_counter)}",
                                        "concept_id" : f"{ontology_name}-{sid}",
                                    }
                                )
                                def_to_syn_counter += 1

    print(f"  *** For {ontology_name}, out of {len(train_samples)},")
    print(f"{round((100*syn_counter)/len(train_samples),2)}% data are FSN_to_SYN ({syn_counter}),")
    print(f"{round((100*def_counter)/len(train_samples),2)}% data are FSN_to_DEF({def_counter}),")
    print(f"{round((100*def_to_syn_counter)/len(train_samples),2)}% data are DEF_to_SYN ({def_to_syn_counter}).\n")

    return train_samples


pre_training_FMA_fspnToSyn = fspn_to_syns(fma_concepts, record_type="FMA", ontology_name="FMA" , USE_DEF=False)
pre_training_NCIT_fspnToSyn = fspn_to_syns(ncit_concepts, record_type="NCIT", ontology_name="NCIT" , USE_DEF=False)
pre_training_SNOMED_fspnToSyn = fspn_to_syns(snomed_concepts, record_type="SNOMED", USE_DEF= True )


pre_train_combined_alltasks_FMA = pre_training_FMA_tree_to_name + pre_training_FMA_fspnToSyn 
pre_train_combined_alltasks_NCIT = pre_training_NCIT_tree_to_name + pre_training_NCIT_CtoP + pre_training_NCIT_hierarchicalId_syns  + pre_training_NCIT_fspnToSyn
pre_train_combined_alltasks_SNOMED =  pre_training_SNOMED_tree_to_name + pre_training_SNOMED_CtoP + pre_training_SNOMED_hierarchicalId_syns  + pre_training_SNOMED_fspnToSyn

pre_train_combined_alltasks_tree = pre_train_combined_alltasks_NCIT + pre_train_combined_alltasks_FMA + pre_train_combined_alltasks_SNOMED


random.shuffle(pre_train_combined_alltasks_tree)

print(f"Percent of each ontology in our dataset: \n SNOMED: {round((100*len(pre_train_combined_alltasks_SNOMED))/len(pre_train_combined_alltasks_tree),2)}% \n NCIT: {round((100*len(pre_train_combined_alltasks_NCIT))/len(pre_train_combined_alltasks_tree),2)}% \n FMA: {round((100*len(pre_train_combined_alltasks_FMA))/len(pre_train_combined_alltasks_tree),2)}% ")

print(f"total Data: {len(pre_train_combined_alltasks_tree)}")




output_dir = f"{DATA_DIR}/pretraining/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dumpjson(os.path.join(output_dir, 'pre_train_paperversion_ontologies.jsonl'), pre_train_combined_alltasks_tree)

print("saved pre-training data as: ",os.path.join(output_dir, 'pre_train_paperversion_ontologies.jsonl'))
