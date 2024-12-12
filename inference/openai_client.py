import openai
from openai import AzureOpenAI
import torch
from inference.constants import AZURE_ENDPOINTS, AZURE_API_VERSION, MODEL_AND_API_KEY_BY_TYPE_AND_REGION, TXT_ONLY, MULTIMOD, US_CODE


class OpenAIClient(object):
    def __init__(self, modality=TXT_ONLY, region=US_CODE, model_type=None, error_file_path=""):
        self.region = region # "us" or other
        self.modality = modality # "text" or "multimodal"
        self.model_type = model_type # can be "mistral", "gpt4o" or None
        self.index = 0
        self.len_gpt_keys = len(MODEL_AND_API_KEY_BY_TYPE_AND_REGION[self.get_key_type()][self.region])
        self.client = self.get_client()
        self.error_file_path = error_file_path + "malfunctioning_api_keys.txt"

    def get_key_type(self):
        return self.model_type if self.model_type is not None else self.modality

    def get_client(self):
        return AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINTS[self.modality][self.region],
            api_version=AZURE_API_VERSION,
            api_key=MODEL_AND_API_KEY_BY_TYPE_AND_REGION[self.get_key_type()][self.region][self.index][1],
        )
    
    def get_modality(self, messages):
        return MULTIMOD if isinstance(messages[0]["content"], list) else TXT_ONLY
    
    def check_modality(self, messages):
        this_modality = self.get_modality(messages)
        if not this_modality == self.modality:
            print(f"Warning: Mismatch in modality between messages and client. Expected {self.modality}, got {this_modality}.")
            print(f"Switching to {this_modality}.")
            self.client.azure_endpoint = AZURE_ENDPOINTS[self.modality][self.region]
            self.modality = this_modality
    
    def move_to_next_api_key(self):
        print("Switching keys")
        self.index = (self.index + 1) % self.len_gpt_keys
        self.client.api_key = MODEL_AND_API_KEY_BY_TYPE_AND_REGION[self.get_key_type()][self.region][self.index][1]

    def generate_text(self, messages, top_logprobs=None):
        self.check_modality(messages)
        response = self.client.chat.completions.create(
            messages=messages,
            model=MODEL_AND_API_KEY_BY_TYPE_AND_REGION[self.get_key_type()][self.region][self.index][0], #Must fill in, optional: gpt-35-turbo、gpt-4、gpt-4-32k
            logprobs=top_logprobs is not None,
            top_logprobs=top_logprobs,
        )
        if top_logprobs is None:
            answer = response.choices[0].message.content
            return answer
        else:
            # returning top_logprobs if computing VQA Score
            is_generated = False
            for top_logprob in response.choices[0].logprobs.content[0].top_logprobs:
                if top_logprob.token.lower().strip().startswith("yes"):
                    is_generated = True
                    return torch.Tensor([top_logprob.logprob]).exp()
            if not is_generated:
                print(f"Warning: answer not generated for VQA Score computation with GPT-4o.")
                print(response.choices[0].logprobs.content[0].top_logprobs)
                return torch.Tensor([0.0])
    
    def get_answer(self, messages, top_logprobs=None):
        made_error = True
        while made_error:
            try:
                answer = self.generate_text(messages, top_logprobs=top_logprobs)
                made_error = False
            except Exception as e:
                if type(e) is openai.BadRequestError:
                    if "The response was filtered due to the prompt triggering Azure OpenAI's content management policy." in str(e):
                        made_error = False
                        answer = ""
                        print("Filtered request")
                    else:
                        print("Bad Request error:", e)
                else:
                    if type(e) is openai.RateLimitError:
                        print("Rate limit error")
                    else:
                        print("Non-Rate limit error:", e)
                        with open(self.error_file_path, 'a') as file:
                            file.write(f"Malfunctioning key: {self.client.api_key}\nAssociated with the following error message:\n{e}\n")
                    self.move_to_next_api_key()
        return answer
