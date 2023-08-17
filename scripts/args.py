from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import cpu_count

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=cpu_count()//2,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )
    # For masked language modeling:
    mlm: bool = field(
        default=False,
        metadata={"help": "Whether to use masked language modeling or not."},
    )
    mlm_mask_probability: float = field(
        default=0.15,
        metadata={
            "help": "The probability of replacing a token with the masked token in the masked language modeling task."
        },
    )
    mlm_max_mask_probability: Optional[float] = field(
        default=None,
        metadata={
            "help": "The maximum rate of the probability of replacing a token with the masked token in the masked "
            "language modeling task. If set to 0, the probability will be set to the value of "
            "`mlm_mask_probability`."
        },
    )
    mlm_mask_increase_epoch_end: Optional[int] = field(
        default=None,
        metadata={
            "help": "The epoch at which the mask probability will be increased from `mlm_mask_probability`."
            "to `mlm_max_mask_probability`."
        },
    )
    mlm_mask_token: Optional[str] = field(
        default="X",
        metadata={"help": "The token to use as a mask for the masked language modeling task."},
    )
    child_parent_separator: Optional[str] = field(
        default="->",
        metadata={
            "help": "The separator to use between the parent and the child in the graph masked language modeling task."
        },
    )
    mlm_mask_at_least_one: bool = field(
        default=False,
        metadata={"help": "Whether to mask at least one token in the masked language modeling task."},
    )
    mlm_only_mask_parents: bool = field(
        default=False,
        metadata={"help": "Whether to only mask the parents of the nodes in the masked language modeling task."},
    )
    mlm_treeid_masking: bool = field(
        default=True,
        metadata={"help": "Whether to mask the treeid in the masked language modeling task. i.e. 0-12-41-X-2-X-0"},
    )
    split_train_to_eval_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "If set, will split the training data into training and evaluation data. The ratio of the "
            "evaluation data will be `split_train_to_eval_ratio`."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")

        # if self.train_file is not None:
        #     extension = self.train_file.split(".")[-1]
        #     # assert extension == "jsonl", "`train_file` should be a json file."
        # if self.validation_file is not None:
        #     extension = self.validation_file.split(".")[-1]
        #     # assert extension == "jsonl", "`validation_file` should be a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

