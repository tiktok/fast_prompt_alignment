# Example command:
# python3 -m data_utils.get_noun_chunks --dataset 100_test_prompts

import argparse
from tqdm import tqdm
import pandas as pd
import os
from data_utils.constants import INPUT_DIR_PATH, OUTPUT_DIR_PATH
from inference.openai_client import OpenAIClient
from inference.constants import TXT_ONLY, US_CODE

def get_noun_chunks(test_source, output_filename):
    df_lines = []
    client = OpenAIClient(modality=TXT_ONLY, region=US_CODE, model_type="gpt4o")
    existing_len = 0
    if os.path.exists(output_filename):
        existing_df = pd.read_csv(output_filename)
        existing_len = existing_df.shape[0]
        df_lines = existing_df.to_dict('records')
    for idx, data in tqdm(enumerate(test_source[existing_len:]), desc=f"Lines processed from {existing_len}-th line"):
        prompt = data.strip()
        messages = [{"role": "user", "content": f"""Decompose the following sentence into individual noun phrases. Ignore prefixes such as "a photo of", "a picture of", "a portrait of", etc. Your response should only be a list of comma-separated values, eg: "foo, bar, baz"\n\n{prompt}"""}]
        made_error = True
        while made_error:
            try:
                noun_chunks = client.generate_text(messages)
                made_error = False
            except Exception as e:
                print("Error", e)
                made_error = True
        df_lines.append({"prompt": prompt, "noun_chunks": noun_chunks})
        if idx % 1000 == 0:
            df = pd.DataFrame.from_dict(df_lines)
            df.to_csv(output_filename[:-4] + f"_{idx}.csv")
    
    df = pd.DataFrame.from_dict(df_lines)
    df.to_csv(output_filename)

def run(filename):
    test_source = open(f"{INPUT_DIR_PATH}{filename}.txt").readlines()
    filename_no_ext = filename.split(".")[0]
    output_filename = f"{OUTPUT_DIR_PATH}{filename_no_ext}_noun_chunks.csv"
    print(filename_no_ext)
    get_noun_chunks(test_source, output_filename)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Script to process data")

    # Add the datapath argument
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed datapath argument
    run(args.dataset)