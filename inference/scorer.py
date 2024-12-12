import os
from inference.constants import AVAILABLE_SCORES, SCORE_SUM_COLUMN, ANNOTATION_SUB_FOLDER, IMAGE_COLUMN
from inference.utils import encode_image, replace_punctuation_and_spaces
from torchvision.transforms.functional import to_pil_image
from diffusers import StableDiffusion3Pipeline
import numpy as np


class Scorer:
    def __init__(self, metrics, output_data_path, dataset_name, client_gpt4o, image_model_pipeline):
        self.clip_score_fn = None
        self.dataset_name = dataset_name
        self.metrics = [score for score in AVAILABLE_SCORES if score in metrics.lower()]
        self.output_data_path = output_data_path
        self.client_gpt4o = client_gpt4o
        self.image_model_pipeline = image_model_pipeline

    @property
    def all_metrics(self):
        return self.metrics + [SCORE_SUM_COLUMN]

    @property
    def all_metrics_and_paraphrase(self):
        return self.all_metrics + ["paraphrase"]
    
    def get_image_path(self, paraphrase, local_process_index, save_for_annotation=False):
        truncated_paraphrase = replace_punctuation_and_spaces(paraphrase)[:50]
        if save_for_annotation:
            return f"{self.output_data_path}{ANNOTATION_SUB_FOLDER}/{self.dataset_name}_{truncated_paraphrase}.png"
        else:
            return f"{self.output_data_path}{self.dataset_name}_process{local_process_index}_{truncated_paraphrase}.png"

    def generate_and_save_image(self, paraphrase, image_path):
        # Generate image from the best paraphrase
        if isinstance(self.image_model_pipeline, StableDiffusion3Pipeline):
            image_output = self.image_model_pipeline(prompt=paraphrase, negative_prompt="", num_images_per_prompt=1, num_inference_steps=28, height=256, width=256).images[0]
            pil_image = image_output
        else:
            image_output = self.image_model_pipeline([paraphrase], num_images_per_prompt=1)
            pil_image = to_pil_image(image_output)
        pil_image.save(image_path)

    def compute_tifa_score_with_gpt4o(self, questions, choices, answers, image_path):
        num_qca = len(questions)
        # Upload image and get URL
        image_url = encode_image(image_path)
        correct_answers = 0
        llm_answers = []
        for qidx in range(num_qca):
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": f"The following is a question about the image below. It is followed by a list of answer choices. You should reply ONLY by copying verbatim one of the answer choices.\nQuestion: {questions[qidx]}\nChoices: {choices[qidx]}\nHere is the image:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                },
            ]
            client_answer = self.client_gpt4o.get_answer(messages)
            llm_answers.append(client_answer)

            correct_answers = correct_answers + int(client_answer.strip().lower() == answers[qidx].strip().lower())

        if num_qca > 0:
            tifa_score = correct_answers/num_qca
        else:
            tifa_score = 0.0

        return tifa_score, llm_answers


    def compute_vqa_score_with_gpt4o(self, noun_chunks, image_path, top_logprobs=None):
        num_chunks = len(noun_chunks)
        # Upload image and get URL
        image_url = encode_image(image_path)
        correct_answers = 0
        probs = []
        for noun_chunk in noun_chunks:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Does this image show \"{noun_chunk}\"? Please answer by giving ONLY yes OR no, and absolutely nothing else."
                        },
                    ]
                },
            ]
            if top_logprobs is None:
                yes_or_no = False
                while not yes_or_no:
                    client_answer = self.client_gpt4o.get_answer(messages).strip().lower()
                    yes_or_no = (client_answer == "yes" or client_answer == "no")
                correct_answers = correct_answers + int(client_answer == "yes")
            else:
                prob = self.client_gpt4o.get_answer(messages, top_logprobs=top_logprobs)
                probs.append(prob.item())
            
        if top_logprobs is None:
            if num_chunks > 0:
                vqa_score = correct_answers/num_chunks
            else:
                vqa_score = 0.0

            return vqa_score
        else:
            return probs


    def get_all_scores(self, prompt_for_image, questions=None, choices=None, answers=None, noun_chunks=None, local_process_index=0, save_for_annotation=False, top_logprobs=None):
        # Generate and save image
        image_path = self.get_image_path(prompt_for_image, local_process_index, save_for_annotation)
        retries = 4
        while not os.path.exists(image_path) and retries > 0:
            self.generate_and_save_image(prompt_for_image, image_path)
            retries -= 1

        sum_score = 0
        scores_dict = {}

        for metric in self.metrics:
            if metric == "tifa":
                # TIFA Score with GPT-4o
                score, llm_answers = self.compute_tifa_score_with_gpt4o(questions, choices, answers, image_path)
                scores_dict["answers"] = llm_answers
            elif metric == "vqa":
                # VQA Score with GPT-4o
                score = self.compute_vqa_score_with_gpt4o(noun_chunks, image_path, top_logprobs=top_logprobs)
                if top_logprobs is not None:
                    scores_dict["vqa_scores"] = score
                    score = np.mean(score)
            sum_score += score
            scores_dict[metric] = score

        scores_dict[SCORE_SUM_COLUMN] = sum_score

        if save_for_annotation:
            scores_dict[IMAGE_COLUMN] = "./" + image_path[len(self.output_data_path):]
        else:
            # Deleting the image
            if os.path.exists(image_path):
                os.remove(image_path)

        return scores_dict
