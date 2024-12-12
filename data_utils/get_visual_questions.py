# Example command:
# python3 -m data_utils.get_visual_questions --dataset 100_test_prompts

from tqdm import tqdm
import pandas as pd
import os
import argparse
from data_utils.constants import INPUT_DIR_PATH, OUTPUT_DIR_PATH, EXAMPLE_MESSAGES
from inference.openai_client import OpenAIClient
from inference.constants import TXT_ONLY, US_CODE

def get_visual_questions(test_source, output_filename):
    df_lines = []
    client = OpenAIClient(modality=TXT_ONLY, region=US_CODE, model_type="gpt4o")
    existing_len = 0
    if os.path.exists(output_filename):
        existing_df = pd.read_csv(output_filename)
        existing_len = existing_df.shape[0]
        df_lines = existing_df.to_dict('records')
    for idx, data in tqdm(enumerate(test_source[existing_len:]), desc=f"Lines processed from {existing_len}-th line"):
        prompt = data.strip()
        messages = EXAMPLE_MESSAGES + [{
            "role": "user",
            "content": f"Description: {prompt}"
        }]
        made_error = True
        while made_error:
            try:
                visual_questions = client.generate_text(messages)
                made_error = False
            except Exception as e:
                print("Error", e)
                made_error = True
        df_lines.append({
            "prompt": prompt,
            "visual_questions": visual_questions,
        })
        if idx % 1000 == 0:
            df = pd.DataFrame.from_dict(df_lines)
            df.to_csv(output_filename[:-4] + f"_{idx}.csv")
    
    pd.DataFrame.from_dict(df_lines).to_csv(output_filename)
    print("Saved to:", output_filename)


def format_questions(dataset):
    # Load the CSV file into a DataFrame
    error_dicts = []

    df = pd.read_csv(f'{OUTPUT_DIR_PATH}{dataset}_visual_questions.csv')

    # Initialize lists to store the extracted data
    prompts = []
    qca = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        text = row['visual_questions']
        prompt = row['prompt']
        
        # Split the text into parts and extract questions, choices, and answers
        anomaly_found = False
        questions = []
        choices = []
        answers = []
        try:
            question_parts = text.split("Q: ")[1:]  # Ignore the first part before any "Q:"
            for part in question_parts:
                question = part.split("Choices: ")[0].strip()
                choices_text = part.split("Choices: ")[1].split("A: ")[0].strip()
                answer = part.split("A: ")[1].split("\n")[0].strip()

                # Append to the lists
                questions.append(question)
                choices.append(choices_text)
                answers.append(answer)
        except:
            anomaly_found = True
            pass

        prompts.append(prompt)
        qca.append({
            "questions": questions,
            "choices": choices,
            "answers": answers
        })

        if anomaly_found:
            print(dataset,index)
            error_dicts.append({
                "dataset": dataset,
                "index": index,
                "prompt": prompt,
                "text": text
            })

    # Create a new DataFrame with the extracted information
    df_extracted = pd.DataFrame({
        'prompt': prompts,
        'qca': qca,
    })

    # Save DataFrame to JSON object (string)
    json_str = df_extracted.to_json(orient='records')

    # Save to a JSON file
    with open(f"{OUTPUT_DIR_PATH}{dataset}_eval_q_a.json", 'w') as file:
        file.write(json_str)

    output_filename = f"{OUTPUT_DIR_PATH}{dataset}_errors.csv"
    pd.DataFrame(error_dicts).to_csv(output_filename)
    print("Saved to:", output_filename)


def run(dataset, only_format):
    if not only_format:
        test_source = open(f"{INPUT_DIR_PATH}{dataset}.txt").readlines()
        output_filename = f"{OUTPUT_DIR_PATH}{dataset}_visual_questions.csv"
        get_visual_questions(test_source, output_filename)
    format_questions(dataset)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Script to process data")

    # Add the datapath argument
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")

    parser.add_argument('--only_format', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed datapath argument
    run(args.dataset, args.only_format)