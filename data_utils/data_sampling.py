# python3 -m data_utils.data_sampling

import argparse
import os
import random
from tqdm import tqdm
from datasets import load_dataset

def deduplicate_prompts(dataset_name, num_samples, input_folder, output_txt):
    # Load dataset
    dataset = load_dataset(dataset_name)

    # Generate random indices
    range_of_indices = range(len(dataset["train"]))
    random_indices = random.sample(range_of_indices, num_samples)

    # Extract new prompts
    new_prompts = [dataset["train"][random_index]["text"] for random_index in random_indices]

    # Load existing prompts from all .txt files in the input folder
    existing_prompts = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            with open(os.path.join(input_folder, file_name), 'r') as file:
                for line in file:
                    # Replace newlines with spaces and strip any leading/trailing whitespace
                    existing_prompts.append(line.replace('\n', ' ').strip())

    # Deduplicate new prompts
    deduplicate_prompts = []
    for new_prompt in tqdm(new_prompts, desc="Deduplication"):
        while new_prompt in existing_prompts:
            new_prompt = dataset["train"][random.choice(range_of_indices)]["text"]
        deduplicate_prompts.append(new_prompt)

    # Save deduplicated new prompts to a text file
    with open(output_txt, 'w') as file:
        for deduplicate_prompt in deduplicate_prompts:
            # Ensure there are no newline characters in the prompts
            prompt = deduplicate_prompt.replace('\n', ' ').strip()
            file.write(prompt + '\n')

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Deduplicate prompts from a dataset.")
    parser.add_argument("--dataset_name", type=str, default="Geonmo/midjourney-prompts-only", help="The name of the dataset to load.")
    parser.add_argument("--num_samples", type=int, default=42000, help="The number of random samples to extract.")
    parser.add_argument("--input_folder", type=str, default="./data/", help="Path to the folder containing input .txt files with existing prompts.")
    parser.add_argument("--output_txt", type=str, default="./data/25k_midjourney_prompts.txt", help="Path to the output text file to save deduplicated new prompts.")

    args = parser.parse_args()

    # Call the deduplication function with the parsed arguments
    deduplicate_prompts(args.dataset_name, args.num_samples, args.input_folder, args.output_txt)

if __name__ == "__main__":
    main()