# !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

import math
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
from accelerate import Accelerator
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import TrainingArguments
from transformers import deepspeed
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ModelArguments:
    access_token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face access token."},
    )
    model_name_or_path: str = field(default="google/gemma-2b-it")
    new_model_path: str = field(default="/mnt/bd/khalil-workspace/annotated_data_models/")
    new_model_name: str = field(default="gemma-2b-it")
    trust_remote_code: bool = False


@dataclass
class DataArguments:
    data_path: str = field(
        default="../annotated_data/10k_train_instruct_tune_for_gemma.csv", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default="../annotated_data/10k_train_instruct_tune_for_gemma.csv", metadata={"help": "Path to the testing/dev data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_seq_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    dataset_text_field: str = "text"
    remove_unused_columns: bool = True
    evaluation_strategy: str = "steps"  # Evaluate every few steps
    eval_accumulation_steps: int = 1
    prediction_loss_only: bool = True
    eval_steps: int = 100  # Evaluation frequency
    save_strategy: str = "steps"  # Save checkpoint every few steps
    save_steps: int = 100  # Save frequency (should match eval_steps)
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"  # Use evaluation loss to determine the best model
    greater_is_better: bool = False
    save_total_limit: int = 2  # Limit the total number of checkpoints


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics. In this example, we're using perplexity.
    """
    loss = eval_preds.metrics["eval_loss"]
    perplexity = math.exp(loss)
    return {"perplexity": perplexity}


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_bias: str = field(default="none")
    use_lora: bool = True
    use_lora_target_modules: bool = False
    lora_target_modules: Optional[List[str]] = field(
        default=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    # lora_weight_path: str = ""
    # lora_bias: str = "none"
    # q_lora: bool = False


@dataclass
class BitsAndBytesArguments:
    # Deactivate 4-bit precision base model loading
    no_4bit: bool = False

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype: str = field(default="float16")
    
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type: str = field(default="nf4")

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant: bool = False

@dataclass
class SFTArguments:
    # Pack multiple short examples in the same input sequence to increase efficiency
    use_packing: bool = False

    # Load the entire model on the GPU 0
    device_map: Optional[Dict[str, int]] = field(default_factory=lambda: {"": Accelerator().process_index})


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none", use_lora=True):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def finetune():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, BitsAndBytesArguments, SFTArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        bitsandbytes_args,
        sft_args,
    ) = parser.parse_args_into_dataclasses()

    # The model that you want to train from the Hugging Face hub
    model_name = model_args.model_name_or_path

    # Fine-tuned model name
    new_model = model_args.new_model_path + model_args.new_model_name

    # Load dataset (you can process it here)
    train_df = pd.read_csv(data_args.data_path)
    # len_data = len(train_df)
    # threshold = int(len_data*0.1)
    # train_df = train_df[:-threshold]
    train_df = pd.DataFrame(train_df)
    training_dataset = Dataset.from_pandas(train_df)

    eval_df = pd.read_csv(data_args.eval_data_path)
    eval_df = pd.DataFrame(eval_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    use_4bit = not bitsandbytes_args.no_4bit

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bitsandbytes_args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bitsandbytes_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bitsandbytes_args.use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=sft_args.device_map,
        token=model_args.access_token,
        attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=model_args.access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        r=lora_args.lora_r,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=lora_args.lora_target_modules if lora_args.use_lora_target_modules else None,
    )

    # Initialize the trainer with the compute_metrics function
    trainer = SFTTrainer(
        model=model,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        packing=sft_args.use_packing,
        compute_metrics=compute_metrics,  # Add this line
    )

    # Train the model
    trainer.train()

    # Save the best model
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=new_model,
        bias=lora_args.lora_bias,
        use_lora=lora_args.use_lora
    )

    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)


if __name__ == "__main__":
    finetune()