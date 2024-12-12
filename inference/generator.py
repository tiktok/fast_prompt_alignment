import torch
from transformers import pipeline
from dataclasses import asdict
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from byteartist.inference_ldm import ByteArtistConfig, ByteArtistInference
from inference.openai_client import OpenAIClient
from inference.constants import HUGGING_FACE_ACCESS_TOKEN


class Generator:
    def __init__(self, num_paraphrases, text_model_name, image_model_name, accelerator, device_map, mode, gpt4_client, gen_kwargs, hf_access_token=HUGGING_FACE_ACCESS_TOKEN):
        self.accelerator = accelerator
        self.num_paraphrases = num_paraphrases
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.mode = mode
        self.gpt4_client = gpt4_client
        self.hf_access_token = hf_access_token # access_token for Hugging Face
        self.text_gen_pipeline = self.init_text_gen_pipeline(device_map)
        self.image_model_pipeline = self.init_image_model_pipeline()
        self.gen_kwargs = asdict(gen_kwargs)

    @property
    def is_iterative(self):
        return self.mode == "iterative"
    
    @property
    def is_instant(self):
        return self.mode == "instant"
    
    @property
    def is_scoring(self):
        return self.mode == "scoring"
    
    @property
    def is_iterative_for_annotation(self):
        return self.mode == "iterative_for_annotation"

    @property
    def using_chatgpt(self):
        return self.text_model_name.lower() == "chatgpt"
    
    @property
    def qwen_mode(self):
        return "qwen" in self.text_model_name.lower()
    
    @property
    def gemma_mode(self):
        return "gemma" in self.text_model_name.lower()
    
    def init_text_gen_pipeline(self, device_map):
        # Load text generation model
        tokenizer_kwargs = {
            "padding_side": "left",
            "low_cpu_mem_usage": "True"
        }

        if self.using_chatgpt:
            text_gen_pipeline = self.gpt4_client
        else:
            text_gen_pipeline = pipeline('text-generation', model=self.text_model_name, torch_dtype=torch.bfloat16 if self.qwen_mode or self.gemma_mode else torch.float16, trust_remote_code=self.qwen_mode, token=self.hf_access_token, device_map=device_map)

            if not self.qwen_mode:
                text_gen_pipeline.tokenizer.pad_token = text_gen_pipeline.tokenizer.eos_token
        
        return text_gen_pipeline
    
    
    def init_image_model_pipeline(self):
        if self.image_model_name == "byteartist":
            return self.init_byteartist()
        # Load image generation model
        image_model_pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.image_model_name,
            torch_dtype=torch.bfloat16 if self.qwen_mode or self.gemma_mode else torch.float16,
        )
        if self.accelerator is not None:
            image_model_pipeline.to(self.accelerator.device)
        else:
            image_model_pipeline.to("cuda")
        return image_model_pipeline
    
    
    def init_byteartist(self):
        image_model_pipeline = ByteArtistInference(ByteArtistConfig(local_rank=0, device="cuda"))
        return image_model_pipeline


    def generate_paraphrases_with_open_source_llm(self, prompt, iteration_number=0, questions=None, choices=None, answers=None, noun_chunks=None, previous_prompts=None):
        ### 1. GENERATE PARAPHRASES WITH OPEN-SOURCE LLM ### 
        if self.is_instant:
            messages = [{
                "role": "user",
                "content": f"""Generate {self.num_paraphrases} paraphrases of the following text-to-image generation prompt while keeping the semantic meaning: "{prompt}". Respond with each new prompt in a new numbered line, e.g.: 
1. paraphrase 1
2. paraphase 2
...
{self.num_paraphrases}. paraphrase {self.num_paraphrases}
Then, give your prompt a score from 0 (worst) to 10 (best), judging its ability to produce aesthetically pleasing AI-generated images that are faithful to what the original prompt wanted to generate. It is encouraged, though not mandatory, to add details that do not contradict the semantic meaning of the original prompt, so that the image can look more aesthetically pleasing, and the paraphrase should be rewarded for that in your score. Likewise, your score will be lower if the paraphrase adds details that contradict the original prompt. Please generate {self.num_paraphrases} diverse paraphrases, and their scores out of 10. Once you have given a score to each of the {self.num_paraphrases} paraphrases, select the one that has the highest score. If many have the same highest score, choose the first one. Then, repeat this process until your score gets to 10/10.
Finally, when you get the best paraphrase, make your choice evident by copying the best paraphrase like this: <BEST> your best paraphrase </BEST>. Do NOT include ANYTHING other than the best paraphrase between <BEST> and </BEST>: no title or number of the best paraphrase, JUST the best paraphrase."""
            }]
        elif self.is_iterative or self.is_iterative_for_annotation:
            message = f"""You are an expert prompt optimizer for text-to-image models. Text-to-image models take a text prompt as input and generate images depicting the prompt as output. You translate prompts written by humans into better prompts for the text-to-image models. Your answers should be concise and effective.\n"""
            if iteration_number == 0:
                messages = [{
                    "role": "user",
                    "content": message + f"""Generate {self.num_paraphrases} paraphrases of the following text-to-image generation prompt while keeping the semantic meaning: "{prompt}". Respond with each new prompt in a new numbered line, e.g.: 
1. paraphrase 1
2. paraphase 2
...
{self.num_paraphrases}. paraphrase {self.num_paraphrases}
The paraphrases should have a high ability to prompt an image generation model to produce aesthetically pleasing AI-generated images that are faithful to what the original prompt wanted to generate. It is encouraged, though not mandatory, to add details that do not contradict the semantic meaning of the original prompt, so that the image can look more aesthetically pleasing. It should be discouraged for the paraphrase adds details that contradict the original prompt. Please generate {self.num_paraphrases} diverse paraphrases."""
                }]
            elif iteration_number > 0:
                message = message + f"""Your task is to optimize this initial prompt written by a human: "{prompt}". Below are some previous prompts with the consistency of each prompt's visual elements in the generated image via a set of multiple-choice questions. Each multiple-choice question is paired with a choice of answers, the correct answer, and the chosen answer of the visual LLM (answer picked). We also indicate whether the answer picked was correct. We call TIFA score the percentage of correctly answered questions for each prompt and generated image. We also provide a decomposition of the prompts' visual elements. Each element is paired with a score indicating the likelihood of its presence in the generated image. We call the VQA score the average of each visual element's score for a given prompt and generated image. The prompts are arranged in ascending order based on their scores, which range from 0 to 100 (higher is better)."""
                idx = 0
                for entry in sorted(previous_prompts, key=lambda entry: entry["score"], reverse=False):
                    idx += 1
                    message = message + f"""

{idx}. {entry["revised_prompt"]}
Overall score: {int(entry["score"]*100)}
TIFA score: {int(entry["tifa"]*100)}
VQA score: {int(entry["vqa"]*100)}
Evaluation Questions:
"""
                    for q_idx in range(len(questions)):
                        how_answered = "It's the wrong answer." if entry["answers"][q_idx] != answers[q_idx] else "It's the correct answer."
                        message = message + f"""- {questions[q_idx]}. Answer Choices: {"; ".join(choices[q_idx])}. Correct Answer: {answers[q_idx]}. Answer picked: {entry["answers"][q_idx]}. {how_answered}\n"""
                    message = message + f"""Visual Elements:\n"""
                    for v_idx in range(len(noun_chunks)):
                        message = message + f"""- {noun_chunks[v_idx]}. Score: {int(entry["vqa_scores"][v_idx]*100)}\n"""
                message = message + f"""Generate {self.num_paraphrases} paraphrases of the initial prompt which keep the semantic meaning and that have higher scores than all the prompts above. Focus on optimizing for the visual elements that are not consistent, such that the resulting VQA and TIFA scores are higher. Favor substitutions and reorderings over additions. Respond with each new prompt in a new numbered line, e.g.: 
1. paraphrase 1
2. paraphase 2
...
{self.num_paraphrases}. paraphrase {self.num_paraphrases}
The paraphrases should have a high ability to prompt an image generation model to produce aesthetically pleasing AI-generated images that are faithful to what the original prompt wanted to generate. It is encouraged, though not mandatory, to add details that do not contradict the semantic meaning of the original prompt, so that the image can look more aesthetically pleasing. It should be discouraged for the paraphrase to contain details that contradict the original prompt. Please generate {self.num_paraphrases} diverse paraphrases."""
                messages = [{
                    "role": "user",
                    "content": message
                }]

        if isinstance(self.text_gen_pipeline, OpenAIClient):
            llm_response = self.text_gen_pipeline.get_answer(messages)
        else:
            prompt_tokens = self.text_gen_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, padding=True, truncation=True)

            gen_tokens = self.text_gen_pipeline(
                [prompt_tokens],
                generation_kwargs=self.gen_kwargs,
                return_full_text=False,
                max_new_tokens=8192
            )

            llm_response = gen_tokens[0][0]["generated_text"]
        return llm_response


    def select_best_of_or_all_paraphrases_with_gpt4(self, llm_response):
        if self.is_instant:
            ### 2. Use GPT-4 to select the best_paraphrase ###
            messages = [{"role": "user", "content": f"""We have a dialogue between a user and an LLM. The user asks the LLM to write {self.num_paraphrases} paraphrases of a prompt, and then give each of the paraphrases a score out of 10. We asked the LLM to put the best paraphrase between these tags: <BEST> and </BEST>, but we are not sure if it did that. Please read what the LLM replied. Then, your response should ONLY be one line, copying the best paraphrase as chosen by the LLM, without any number indices or indentation. Here is the LLM's reply:\n\"{llm_response}\""""}]
        elif self.is_iterative or self.is_iterative_for_annotation:
            ### 2. Use GPT-4 to select the data_args.num_paraphrases paraphrases ###
            messages = [{"role": "user", "content": f"""We have a dialogue between a user and an LLM. The user asks the LLM to write {self.num_paraphrases} paraphrases of a prompt. Read what the LLM replied. Then, your response should ONLY be {self.num_paraphrases} lines, copying the {self.num_paraphrases} paraphrases generated by the LLM, one paraphrase by line without any number indices or indentation. Here is the LLM's reply:\n\"{llm_response}\""""}]
        client_answer = self.gpt4_client.get_answer(messages)
        if self.is_instant:
            best_paraphrase = client_answer.strip()
            return best_paraphrase
        elif self.is_iterative or self.is_iterative_for_annotation:
            paraphrases = [paraphrase.strip() for paraphrase in client_answer.strip().split("\n") if len(paraphrase.strip()) > 0]
            # assert len(paraphrases) == data_args.num_paraphrases
            return paraphrases
        

    def get_best_of_or_all_paraphrases(self, prompt, iteration_number=0, questions=None, choices=None, answers=None, noun_chunks=None, previous_prompts=None):
        # 1. GENERATE PARAPHRASES WITH OPEN-SOURCE LLM
        llm_response = self.generate_paraphrases_with_open_source_llm(prompt, iteration_number, questions, choices, answers, noun_chunks, previous_prompts)

        # 2. Use GPT-4 to get best of or all paraphrases
        return self.select_best_of_or_all_paraphrases_with_gpt4(llm_response)


