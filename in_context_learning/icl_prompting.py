import pandas as pd
from inference.openai_client import OpenAIClient  # Assuming the OpenAI client is in the inference package
import os
from tqdm import tqdm

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Build the path to the "data_prepped" subfolder
data_prepped_path = os.path.join(parent_dir, "data_prepped")

class PromptImprover:
    def __init__(self, model: str = "mistral"):
        self.client = OpenAIClient(model_type=model)

    def load_in_context_examples(self, file_path: str):
        """
        Load the in-context examples dataset containing user and gold prompts.
        Assumes the dataset has columns 'user_prompt' and 'gold_prompt'.
        """
        return pd.read_csv(file_path)

    def load_test_prompts(self, file_path: str):
        """
        Load the test dataset containing user prompts.
        Assumes the dataset has a column 'user_prompt'.
        """
        return pd.read_csv(file_path)

    def generate_improved_prompt(self, user_prompt: str, examples: pd.DataFrame) -> str:
        """
        Generate an improved prompt for a given user prompt using in-context learning examples.
        A random sample of 100 examples from the dataset will be used.
        """
        # Sample approximately 100 in-context examples
        # sampled_examples = examples.sample(n=min(100, len(examples)))

        # Construct the prompt to the model by concatenating the sampled examples
        in_context_prompt = [{
            "role": "system",
            "content": "You are a prompt improver for a text-to-image generation model. You are improving prompts in a way that is specific to one such model, and you are expected to improve the prompts in a way that is specific to that model, such that the images are faithful to the original user prompt, and more aesthetically pleasing and complete than if they had been generated without any prompt improver."
        }]
        for _, row in sampled_examples.iterrows():
            in_context_prompt += [{
                "role": "user",
                "content": f"User prompt: {row['prompt']}"
            }, {
                "role": "assistant",
                "content": f"Improved prompt: {row['best_paraphrase']}"
            }]
        # Add the user prompt for which we want to generate the improved version
        full_prompt = in_context_prompt + [{
            "role": "user",
            "content": f"User prompt: {user_prompt}"
        }]

        # Call the OpenAI API client to generate the improved prompt
        response = self.client.get_answer(full_prompt)

        return response

    def improve_prompts_in_test_set(self, in_context_file: str, test_file: str, output_file: str):
        """
        Generate improved prompts for all user prompts in the test dataset.
        Saves the result (user_prompt, improved_prompt) to a CSV file.
        """
        # Load the in-context learning examples and the test prompts
        in_context_examples = self.load_in_context_examples(in_context_file)
        test_prompts = self.load_test_prompts(test_file)

        # Initialize a list to store results
        results = []

        # For each test user prompt, generate the improved prompt
        for _, row in tqdm(test_prompts.iterrows(), desc=test_file):
            user_prompt = row['prompt']
            improved_prompt = self.generate_improved_prompt(user_prompt, in_context_examples)

            # Append the result
            results.append({'prompt': user_prompt, 'improved_prompt': improved_prompt})

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

# Example usage:
if __name__ == "__main__":
    improver = PromptImprover()

    # Paths to your datasets and output file
    in_context_file = f"{data_prepped_path}/original_descriptions_42k_best_paraphrase.csv"
    datasets = ["partiprompt_best_paraphrase", "coco_captions_best_paraphrase"]
    for dataset in datasets:
        print("dataset:", dataset)
        test_file = f"{data_prepped_path}/{dataset}.csv"
        output_file = f"mistral_large_results_{dataset}_no_icl_examples.csv"

        # Generate improved prompts for the test dataset
        improver.improve_prompts_in_test_set(in_context_file, test_file, output_file)