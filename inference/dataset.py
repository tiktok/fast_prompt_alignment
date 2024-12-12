import pandas as pd
import os

class Dataset:
    def __init__(self, eval_data_path, dataset_name, starting_point, ending_point, scoring_column):
        self.name = dataset_name
        dataframe, self.dataframe_type = self.get_dataframe(eval_data_path, dataset_name, scoring_column)
        self.rows = list(dataframe.iterrows())
        self.starting_point = starting_point
        self.ending_point = len(self.rows) if ending_point == -1 or ending_point < starting_point else ending_point
        self.scoring_column = scoring_column

    def get_dataframe(self, eval_data_path, dataset_name, scoring_column):
        # Load the CSV files into DataFrames
        question_df_path = os.path.join(eval_data_path, f"{dataset_name}_eval_q_a.json")
        if os.path.exists(question_df_path):
            question_df = pd.read_json(question_df_path, orient='records')
        else:
            question_df = None
        
        noun_chunk_df_path = os.path.join(eval_data_path, f"{dataset_name}_noun_chunks.csv")
        if os.path.exists(noun_chunk_df_path):
            noun_chunk_df = pd.read_csv(noun_chunk_df_path)
        else:
            noun_chunk_df = None

        scoring_df = None
        if len(scoring_column) > 0:
            scoring_path = os.path.join(eval_data_path, f"{dataset_name}_{scoring_column}.csv")
            if os.path.exists(scoring_path):
                scoring_df = pd.read_csv(scoring_path)

        if noun_chunk_df is not None and question_df is not None:
            # Merge the DataFrames on common columns
            merged_df = question_df.merge(noun_chunk_df, how='inner', on="prompt")
            if scoring_df is not None:
                merged_df = merged_df.merge(scoring_df, how='inner', on="prompt")
            return merged_df, "MERGED"
            # columns: ['Unnamed: 0', 'prompt', 'qca', 'noun_chunks']
        elif noun_chunk_df is not None:
            return noun_chunk_df, "NOUN_CHUNK"
        elif question_df is not None:
            return question_df, "QCA"
        else:
            raise ValueError("Both noun_chunk_df and question_df are None")
    
    def get_info_from_row(self, row):
        original_prompt = row["prompt"].strip()

        noun_chunks = None
        if not self.dataframe_type == "QCA":
            # Noun Chunks for dCS scoring
            noun_chunks = [noun_chunk.strip() for noun_chunk in str(row["noun_chunks"]).split(',')]

        questions, choices, answers = None, None, None
        if not self.dataframe_type == "NOUN_CHUNK":
            # Questions, Choices, Answers for TIFA scoring
            questions = row["qca"]["questions"]
            choices = row["qca"]["choices"]
            answers = row["qca"]["answers"]

        return [original_prompt, questions, choices, answers, noun_chunks]