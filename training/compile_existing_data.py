from collections import defaultdict
import re
import pandas as pd
import os
from inference.constants import ORIGINAL_COLUMN
from transformers import HfArgumentParser
from inference.arguments import *

def compile_results(output_data_dir, dataset_name, remove=False):
    # Initialize a list to store dataframes
    dfs = []
    base_dir_files = os.listdir(output_data_dir)
    these_files = [f for f in base_dir_files if f.startswith(dataset_name) and f.endswith(".csv")]

    # Extract the final list of filenames with the highest number for each process
    final_filenames = [os.path.join(output_data_dir, filename) for filename in these_files if len(filename) > 0]
    for file_name in final_filenames:
        if os.path.exists(file_name):
            try:
                df = pd.read_csv(file_name)
                dfs.append(df)
            except:
                print("Error reading file:", file_name)
                # Handle the case, e.g., by creating an empty DataFrame
            if remove:
                os.remove(file_name)

    # If there are dataframes, concatenate them and calculate the averages
    if dfs:
        # Combine all dataframes
        combined_file_name = f"{output_data_dir}{dataset_name}_combined.csv"
        combined_df = pd.concat(dfs)
        combined_df = combined_df.drop_duplicates(subset=[ORIGINAL_COLUMN])
        combined_df.to_csv(combined_file_name, index=False)

def run():
    parser = HfArgumentParser((DataArguments))
    data_args = parser.parse_args_into_dataclasses()[0]

    # Compile results
    compile_results(data_args.output_data_path, data_args.dataset_name, remove=False)


if __name__ == "__main__":
    run()