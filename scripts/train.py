# starting point is https://github.com/huggingface/transformers/blob/master/examples/pytorch/translation/run_translation.py
import json
import logging

import os
import random
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from typing import Any, Optional, Union

import datasets
import numpy as np
import pytz

import transformers
import wandb
from datasets import load_dataset, load_metric

from transformers import (  
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    default_data_collator,
    set_seed,
    get_polynomial_decay_schedule_with_warmup,
    AdamW
)

from transformers.trainer_utils import (  
    EvalLoopOutput,
    EvalPrediction,
    get_last_checkpoint,
)

from args import DataTrainingArguments, ModelArguments
from customTrainer import Seq2SeqTrainer


logger = logging.getLogger(__name__)


logging.disable(logging.WARNING)
datasets.logging.set_verbosity_error()

warnings.filterwarnings("ignore", category=DeprecationWarning)

datasets.logging.set_verbosity_error()

# argparser = argparse.ArgumentParser(scriptArgs)
# script_args = argparser.parse_args()

today = datetime.now(pytz.timezone("US/Pacific")).strftime("%m-%d-%y-%H%M")

evaluation_test_data_file = None




parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()



# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# %%
# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# %%
# Set seed before initializing model.
set_seed(training_args.seed)

# %%
# Get the datasets: you can either provide your own JSON training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For translation, only JSON files are supported, with one field named "translation" containing two keys for the
# source and target languages (unless you adapt what follows).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
else:
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    if evaluation_test_data_file:
        data_files["evaluation_test"] = evaluation_test_data_file
        extension = data_args.test_file.split(".")[-1]

    if extension == "jsonl":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

# Split the datasets into training and validation sets (if needed):
if data_args.split_train_to_eval_ratio:
    if data_args.mlm:
        raw_datasets["validation"] = raw_datasets["train"].train_test_split(data_args.split_train_to_eval_ratio)["test"]
    else:
        raw_datasets_split = raw_datasets["train"].train_test_split(data_args.split_train_to_eval_ratio)
        raw_datasets["validation"] = raw_datasets_split["test"]
        raw_datasets["train"] = raw_datasets_split["train"]


# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# %%
# from transformers import T5ForConditionalGeneration  # , T5Tokenizer

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
model = T5ForConditionalGeneration.from_pretrained(  # AutoModelForSeq2SeqLM.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

model.resize_token_embeddings(len(tokenizer))

# %%
# Preprocessing the datasets.
# We need to tokenize inputs and targets.
if training_args.do_train:
    column_names = raw_datasets["train"].column_names
elif training_args.do_eval:
    column_names = raw_datasets["validation"].column_names
elif training_args.do_predict:
    column_names = raw_datasets["test"].column_names
else:
    raise ValueError("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")

# %%
# Temporarily set max_target_length for training.
max_target_length = data_args.max_target_length
padding = "max_length" if data_args.pad_to_max_length else False


# %%


def mlm_preprocess_function(samples):

    ontology_ids = {
        "SNOMED": "0",
        "FMA": "1",
        "NCIT": "2",
    }

    definition_mappings = {
        "FSN": "F",
        "SYN": "S",
        "DEF": "D",
    }

    def add_prefix(record_type):
        string_list = record_type.split("_")
        if string_list[1] == "TREE":
            return None, definition_mappings[string_list[3]] + ontology_ids[string_list[0]] + ": "
        else:
            return (
                definition_mappings[string_list[1]] + ontology_ids[string_list[0]] + ": ",
                definition_mappings[string_list[3]] + ontology_ids[string_list[0]] + ": ",
            )

    def child_parent_preprocessing(data):
        """Preprocess the chaild to parent inputs.

        Args:
            data: the CtoP data

        Returns:
            string: prepocessed input.
        """

        child = data[0]
        # Parents:
        parents = list()
        for p in data[1]:
            parents.append(p)

        parents = " ".join(parents)

        return child + "|" + parents

    inputs = []
    # targets = []
    types = []
    document_ids = []
    model_inputs = defaultdict()
    # print(samples['text'])
    for sample in samples["text"]:
        j = json.loads(sample)

        # CtoP:
        if j["record_type"] in ["SNOMED_CT_TREE", "SNOMED_TREE_CP", "NCIT_TREE_CP"]:
            j["input"] = child_parent_preprocessing(j["data"])
            prefix = "MLMCtoP:|"

        # SYN:
        elif j["record_type"] in ["SNOMED_CT_SYN_TREE", "SNOMED_SYN_TREE", "NCIT_SYN_TREE"]:

            j["input"] = "|".join(j["data"])
            prefix = "MLMSyn:|"

        # TREE TO FSN, TREE TO SYN, TREE TO Def:
        elif j["record_type"] in [
            "SNOMED_TREE_TO_FSN",
            "FMA_TREE_TO_FSN",
            "NCIT_TREE_TO_FSN",
            "SNOMED_TREE_TO_SYN",
            "FMA_TREE_TO_SYN",
            "NCIT_TREE_TO_SYN",
            "SNOMED_TREE_TO_DEF",
            "FMA_TREE_TO_DEF",
            "NCIT_TREE_TO_DEF",
        ]:
            # j["data"].insert(1, "N:")  # Add the special prefix in front of the text.
            in_data_prefixes = add_prefix(j["record_type"])
            j["data"][1] = in_data_prefixes[1] + j["data"][1]
            j["input"] = "|".join(j["data"])
            prefix = "MLMtreeTOeng:|"

        # FSN TO SYN, FSN TO DEF, DEF TO SYN:
        elif j["record_type"] in [
            "NCIT_FSN_to_SYN",
            "SNOMED_FSN_to_SYN",
            "FMA_FSN_to_SYN",
            "NCIT_FSN_to_DEF",
            "SNOMED_FSN_to_DEF",
            "FMA_FSN_to_DEF",
            "NCIT_DEF_to_SYN",
            "SNOMED_DEF_to_SYN",
            "FMA_DEF_to_SYN"
        ]:
            in_data_prefixes = add_prefix(j["record_type"])
            j["data"][0] = in_data_prefixes[0] + j["data"][0]
            j["data"][1] = in_data_prefixes[1] + j["data"][1]
            j["input"] = "|".join(j["data"])
            prefix = "MLMengTOeng:|"
        
        elif j['record_type'] == "UNLABED_TEXT":
            prefix = "UNLABEL:|"

        else:
            raise ValueError(f"Record type {j['record_type']} is not supported.")

        inputs.append(prefix + j["input"])
        types.append(j["record_type"])


    model_inputs["types"] = types
    # The name has to be something that would not be removed in the trainer (i.e. labels, input_ids, etc.) (we choose labels:)
    model_inputs["labels"] = inputs
    return model_inputs


def preprocess_function(samples):
    inputs = []
    targets = []
    types = []
    document_ids = []
    prompt = ""
    prefix = ""
    # print(samples['text'])
    for sample in samples["text"]:
        j = json.loads(sample)

        prompt = ""
        prefix = ""

        if j["record_type"] in ["SNOMED_CT","SNOMED_TREE_TO_FSN", "SNOMED_TREE_TO_SYN", "SNOMED_TREE_TO_DEF","SNOMED_FSN_tree:F0-F0"]:
            prefix = "F0: "
            prompt = " 0-"
            j["output"] = j["output"][2:]

        elif j["record_type"] == "SNOMED_CT_GRAPH_CtoP":
            prefix = "CtoP: "  # Child to Parents

        elif j["record_type"] == "SNOMED_CT_GRAPH_PtoC":
            prefix = "PtoC: "  # Parent to Childeten

        elif j["record_type"] == "FMA":
            prefix = "F0: "
            prompt = " F1:"
        elif j["record_type"] == "NCIT":
            prefix = "F0: "
            prompt = " F2:"
        elif j["record_type"] == "FMA:F0-F1":
            prefix = "F0: "
            prompt = " F1:"
        elif j["record_type"] == "FMA:S1-F1":
            prefix = "S0: "
            # prefix = "S1: "
            prompt = " F1:"
        elif j["record_type"] == "FMA:S0-F1":
            prefix = "S0: "
            prompt = " F1:"
        elif j["record_type"] in ["NCIT_PHAR:F0-F2", "NCIT_NEOP:F0-F2"]:
            prefix = "F0: "
            prompt = " F2:"
        elif j["record_type"] in ["NCIT_PHAR:S2-F2", "NCIT_NEOP:S2-F2"]:
            prefix = "S0: "
            # prefix = "S2: "
            prompt = " F2:"
        elif j["record_type"] in ["NCIT_PHAR:S0-F2", "NCIT_NEOP:S0-F2"]:
            prefix = "S0: "
            prompt = " F2:"
        elif j["record_type"] == "FMA_TREEID":
            prefix = "F0: "
        elif j["record_type"] == "NCIT_TREEID":
            prefix = "F0: "
        elif j["record_type"] == "FMA_TREEID_FSN":
            prefix = "FMA: "
        elif j["record_type"] == "FMA_TREEID_FSN_FROM_FSN":
            prefix = "FMA_FSN: "
        elif j["record_type"] == "NCIT_TREEID_FSN":
            prefix = "NCIT: "
        elif j["record_type"] == "NCIT_TREEID_FSN_FROM_FSN":
            prefix = "NCIT_FSN: "
        elif j["record_type"] == "FMA_tree:F0-F1":
            prefix = "F0: "
            prompt = " 1-"
            j["output"] = j["output"][2:]
        elif j["record_type"] in ["FMA_tree", "FMA_tree:S1-F1", "FMA_tree:S0-F1", "FMA_tree:F1-F1"]:
            # prefix = "S0: "
            prefix = "F0: "
            prompt = " 1-"
            j["output"] = j["output"][2:]
        elif j["record_type"] in ["NCIT_PHAR_tree", "NCIT_NEOP_tree", "NCIT_PHAR_tree:F0-F2", "NCIT_NEOP_tree:F0-F2"]:
            prefix = "F0: "
            prompt = " 2-"
            j["output"] = j["output"][2:]
        elif j["record_type"] in [
            "NCIT_NEOP_tree:S2-F2",
            "NCIT_PHAR_tree:S2-F2",
            "NCIT_PHAR_tree:S0-F2",
            "NCIT_NEOP_tree:S0-F2",
            "NCIT_NEOP_tree:F2-F2",
            "NCIT_PHAR_tree:F2-F2",
        ]:

            prefix = "F0: "
            prompt = " 2-"
            j["output"] = j["output"][2:]
        else:
            raise ValueError(f"Record type {j['record_type']} is not supported.")
        inputs.append(prefix + j["input"] + prompt)
        targets.append(j["output"])
        types.append(j["record_type"])
        if "document_id" not in j:
            document_ids.append("fake_doc_id")
        else:
            document_ids.append(j["document_id"])

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.

    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(la if la != tokenizer.pad_token_id else -100) for la in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["types"] = types
    model_inputs["document_ids"] = document_ids

    # model_inputs["true_inputs"] = inputs
    # model_inputs["targets"] = targets
    return model_inputs


def mask_data_collator(
    input,
    tokenizer,
    padding,
    masking_percentage=0.0,
    mask_at_least_one=False,
    only_mask_parents=False,
    mask_token="<X>",
    child_parent_separator="->",
):
    """
    Mask step on the data collator. (batch inputs)
    # Edited to mask at least one token in the input.

    Args:
        input (dict): input['labels'] which is the output of the preprocess_function.
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (bool): Description in CustomDataCollatorForSeq2Seq
        masking_percentage (float, optional): percentage of tokens to mask. Defaults to 0.0.
        mask_at_least_one (bool, optional): mask at least one token in the input.. Defaults to False.
        only_mask_parents (bool, optional): mask only the parents. Defaults to False.
        mask_token (str, optional): The mask token. Defaults to "<X>".
        child_parent_separator (str, optional): The child to parent separator. Defaults to "->".

    Returns:
        dict: the dict for the Dataloader with implemented masking.
    """
    #checks if the string is a treeid 
    def token_is_treeid(string):
        return bool(re.fullmatch(r"\d+(-\d+)+", string))

    def mask_treeid(tree_id):
        masked_id = []
        train_o = []
        for i in tree_id.split("-"):
            if random.random() < masking_percentage:
                train_o.append(i)
                masked_id.append(mask_token)
            else:
                masked_id.append(i)
        masked_id = "-".join(masked_id)
        return masked_id, train_o

    train_i = list()
    train_o = list()

    input_tokens = input["labels"].split("|")
    #other_token serves as the list of tokens that we want to do masking, 

    if input_tokens[0] == "MLMSyn:":
        # train_i.append(input_tokens[0]) # We don't need to have the prefix for these tasks.
        other_tokens = input_tokens[1:]

    elif input_tokens[0] in ["MLMtreeTOeng:", "MLMengTOeng:"]:

        other_tokens = input_tokens[1:]

        # Fillip definition and ID's for 100% of the time:
        if input_tokens[0] == "MLMtreeTOeng:":
            other_tokens.reverse()

        # Fillip definition and SYNS's for 50% of the time
        if random.random() < 0.5 and input_tokens[0] == "MLMengTOeng:":
            other_tokens.reverse()

        other_tokens = " ".join(other_tokens).split(" ")

    elif input_tokens[0] == "MLMCtoP:":
        train_i.append("CP:")  # Make it shorter. | 
        if only_mask_parents:
            # Check for masking the treeid:
            if data_args.mlm_treeid_masking and token_is_treeid(input_tokens[1]):
                masked_id, output_add = mask_treeid(input_tokens[1])
                train_o = train_o + output_add
                train_i.append(masked_id)
            else:
                train_i.append(input_tokens[1])
            other_tokens = input_tokens[2].split(" ")
        else:
            other_tokens = [input_tokens[1]] + input_tokens[2].split(" ")
    # print(train_i)
    # print(other_tokens)


    elif input_tokens[0] in ["UNLABEL:" ]:
        train_i.append("UL:") 
        other_tokens = re.split(r"\s", input_tokens[1])
    
    # NO MLM: (other tokens are empty so no masking)
    elif input_tokens[0] in ["SNOMEDSPAN:" ]:
        train_i.append("M:") # SSR ( treeid predict) 
        train_i.append(input_tokens[1])
        other_tokens = []
        train_o = input_tokens[2:]
    

    else:
        raise ValueError(f"Record type {input_tokens[0]} is not supported.")

    masked_i = -1
    if mask_at_least_one: #grantee that at list one token will get masked
        num_tokens = len(other_tokens)
        masked_i = random.randint(0, num_tokens - 1)


    for i, token in enumerate(other_tokens):
        if token == child_parent_separator: # is not executing 
            train_i.append(token)
        elif re.match(r"^\S+:$", token): #<text>: -> not masking <text> , pre
            train_i.append(token)
        elif i == masked_i:
            train_i.append(mask_token)
            train_o.append(token)
        else:
            if random.random() < masking_percentage:
                train_o.append(token)
                train_i.append(mask_token)
            elif data_args.mlm_treeid_masking and token_is_treeid(token):  # Applying masked on treeid
                masked_id, output_add = mask_treeid(token)
                train_o = train_o + output_add
                train_i.append(masked_id)
            else:
                train_i.append(token)

    data = {
        "input": " ".join(train_i),
        "output": " ".join(train_o),
    }
    # print( "input: ", data['input'])
    # print("output: ",  data['output'])

    model_inputs = tokenizer(data["input"], max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(data["output"], max_length=data_args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.

    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(la if la != tokenizer.pad_token_id else -100) for la in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


@dataclass
class CustomDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    # Custom args:
    data_size: int = None
    data_counter: int = 0

    def __call__(self, features, return_tensors=None):
        # print(features)
        # Check if we need to increase the masking percentage:
        self.data_counter += len(features)

        if data_args.mlm_max_mask_probability and data_args.mlm_mask_increase_epoch_end:
            # Lets make it linear per batch:
            data_covered_percentage = self.data_counter / (self.data_size * data_args.mlm_mask_increase_epoch_end)
            data_args.masking_percentage = (
                data_args.mlm_mask_probability
                + (data_args.mlm_max_mask_probability - data_args.mlm_mask_probability) * data_covered_percentage
            )
        else:
            data_args.masking_percentage = data_args.mlm_mask_probability

        # Masking:
        features = [
            mask_data_collator(
                data,
                tokenizer,
                padding=padding,
                masking_percentage=data_args.masking_percentage,
                mask_at_least_one=data_args.mlm_mask_at_least_one,
                mask_token=data_args.mlm_mask_token,
                child_parent_separator=data_args.child_parent_separator,
                only_mask_parents=data_args.mlm_only_mask_parents,
            )
            for data in features
        ]

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(la) for la in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
    logger.warning(
        "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
        f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
    )

if training_args.do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    # SHUFFLING TRAINING DATASET
    train_dataset = train_dataset.shuffle(seed=42)

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            mlm_preprocess_function if data_args.mlm else preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
if training_args.do_eval:
    max_target_length = data_args.val_max_target_length
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            mlm_preprocess_function if data_args.mlm else preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

if training_args.do_predict:
    max_target_length = data_args.val_max_target_length
    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test"]
    if evaluation_test_data_file:
        evaluation_predict_dataset = raw_datasets["evaluation_test"]
    if data_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        evaluation_predict_dataset = evaluation_predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on evaluation prediction dataset",
        )
# %%
# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
if data_args.mlm:
    print("load CustomDataCollatorForSeq2Seq")
    data_collator = CustomDataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        data_size=len(train_dataset),
    )
else:
    if data_args.pad_to_max_length:
        print("load default_data_collator")
        data_collator = default_data_collator
    else:
        print("load DataCollatorForSeq2Seq")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )


training_args.generation_max_length = 1024


# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    post_process_function=post_processing_function if training_args.predict_with_generate else None,
)

# %%
# Training
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
results = {}
max_length = (
    training_args.generation_max_length
    if training_args.generation_max_length is not None
    else data_args.val_max_target_length
)
num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

if training_args.do_eval:
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

