import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import transformers
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator

@dataclass
class ModelArguments:
    model_name: str = field(default="NousResearch/Llama-2-7b-chat-hf")
    model_path: str = field(default="./model")


@dataclass
class DataArguments:
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    eval_data_column_name: str = field(
        default=None, metadata={"help": "Name of the column containing original user prompts."}
    )
    output_data_path: str = field(
        default=None, metadata={"help": "Path to the output data."}
    )

def generate():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    (
        model_args,
        data_args,
    ) = parser.parse_args_into_dataclasses()

    model_name = model_args.model_name
    device_map = {"": Accelerator().process_index}
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    prompt_df = pd.read_csv(data_args.eval_data_path)
    prompts = prompt_df[data_args.eval_data_column_name].tolist()[:200]

    data_dicts = []
    
    idx = 0
    batch_size = 4
    prompt_total = len(prompts)
    iterations = prompt_total // batch_size + int(prompt_total % batch_size > 0)
    for _ in tqdm(range(iterations)):
        these_prompts = prompts[idx:idx+batch_size]
        sens = [f"<s> [INST] Rephrase the following text-to-image generation prompt to make it extremely detailed and descriptive. The rephrased prompt should be faithful to the original prompt's intent, clear and fluent, and creative without contradicting the original prompt. Here is the original prompt: {prompt} [/INST] Absolutely! Here is the rephrased prompt:" for prompt in these_prompts]

        prompt_tokens = tokenizer(sens, return_tensors='pt', padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            gen_tokens = base_model.generate(
                **prompt_tokens,
                do_sample=True,
                num_beams=8,
                max_new_tokens=512,  # Reduce token length if possible
                temperature=0.1,  # Adjust temperature for more creative responses
                top_p=0.9,  # Nucleus sampling for diversity
                no_repeat_ngram_size=3,  # Reduce repetitive sequences
            )
        gen_text = [text.split("Here is the rephrased prompt:")[1].strip() for text in tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)]
        for original_prompt, generated_prompt in zip(these_prompts, gen_text):
            data_dicts.append({
                "original_prompt": original_prompt,
                "generated_prompt": generated_prompt,
            })
        idx += batch_size

    output_df = pd.DataFrame(data_dicts)
    output_df.to_csv(data_args.output_data_path, index=True)

if __name__ == "__main__":
    generate()