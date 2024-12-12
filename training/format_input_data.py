import pandas as pd
import json
from dataclasses import dataclass, field
import transformers


@dataclass
class DataArguments:
    input_file_path: str = field(
        default="../annotated_data/10k_train_with_rephrase.csv", 
        metadata={"help": "Path to the evaluation data."}
    )
    original_prompt_column: str = field(
        default="original_prompt", 
        metadata={"help": "Name of the column containing original prompts."}
    )
    rephrased_prompt_column: str = field(
        default="rephrased_prompt", 
        metadata={"help": "Name of the column containing rephrased prompts."}
    )
    corrected_prompt_column: str = field(
        default=None, 
        metadata={"help": "Name of the column containing rephrased and corrected prompts."}
    )
    output_file_path: str = field(
        default="../annotated_data/10k_train_instruct_tune_for_hugging_face.csv", 
        metadata={"help": "Path to the output data."}
    )
    format: str = field(
        default="hf", 
        metadata={"help": "Name of the format wanted: hf (Hugging Face), qwen, gemma, chatml, test."}
    )


def convert_to_inference_train_format():
    parser = transformers.HfArgumentParser(
        (DataArguments)
    )
    (
        data_args,
    ) = parser.parse_args_into_dataclasses()

    input_file_path = data_args.input_file_path
    output_file_path = data_args.output_file_path

    df = pd.read_csv(input_file_path)
    original_prompts = list(df.to_dict()[data_args.original_prompt_column].values())
    rephrased_prompts = list(df.to_dict()[data_args.rephrased_prompt_column].values())
    if data_args.corrected_prompt_column is not None:
        corrected_prompts = list(df.to_dict()[data_args.corrected_prompt_column].values())
        rephrased_corrected_prompts = []
        for rephrased_prompt, corrected_prompt in zip(rephrased_prompts, corrected_prompts):
            if corrected_prompt == "Correct.":
                rephrased_corrected_prompts.append(rephrased_prompt)
            else:
                rephrased_corrected_prompts.append(corrected_prompt)
        rephrased_prompts = rephrased_corrected_prompts

    output_dict = []
    for original_prompt, rephrased_prompt in zip(original_prompts, rephrased_prompts):
        if data_args.format == "gemma":
            this_dict = {
                "text": "<start_of_turn>user Rephrase the following text-to-image generation prompt to make it extremely detailed and descriptive. The rephrased prompt should be faithful to the original prompt's intent, clear and fluent, and creative without contradicting the original prompt. Here is the original prompt: " + original_prompt + "<end_of_turn> <start_of_turn>model Absolutely! Here is the rephrased prompt: " + rephrased_prompt,
            }
        elif data_args.format == "qwen":
            this_dict = [{
                "from": "user",
                "value": "Rephrase the following text-to-image generation prompt to make it extremely detailed and descriptive. The rephrased prompt should be faithful to the original prompt's intent, clear and fluent, and creative without contradicting the original prompt. Here is the original prompt: " + original_prompt
            }, {
                "from": "assistant",
                "value": "Absolutely! Here is the rephrased prompt: " + rephrased_prompt
            }]
        elif data_args.format == "chatml":
            this_dict = {
                "text": "<|im_start|>user Rephrase the following text-to-image generation prompt to make it extremely detailed and descriptive. The rephrased prompt should be faithful to the original prompt's intent, clear and fluent, and creative without contradicting the original prompt. Here is the original prompt: " + original_prompt + "<|im_end|> <|im_start|>assistant Absolutely! Here is the rephrased prompt: " + rephrased_prompt,
            }
        elif data_args.format == "hf":
            this_dict = {
                "text": "<s> [INST] Rephrase the following text-to-image generation prompt to make it extremely detailed and descriptive. The rephrased prompt should be faithful to the original prompt's intent, clear and fluent, and creative without contradicting the original prompt. Here is the original prompt: " + original_prompt + " [/INST] Absolutely! Here is the rephrased prompt: " + rephrased_prompt,
            }
        elif data_args.format == "test":
            this_dict = {
                "original_prompt": original_prompt,
                "rephrased_and_corrected_prompt": rephrased_prompt
            }
        else:
            raise NotImplementedError("Choose from the following formats: hf, qwen, gemma, test, chatml")
        output_dict.append(this_dict)

    if data_args.format == "qwen":
        out_file = open(output_file_path, "w")
        json.dump(output_dict, out_file)
        out_file.close()
    else:
        output_df = pd.DataFrame(output_dict)
        output_df.to_csv(output_file_path)


if __name__ == "__main__":
    convert_to_inference_train_format()