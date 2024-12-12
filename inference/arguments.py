from dataclasses import dataclass, field
from inference.constants import US_CODE

@dataclass
class ModelArguments:
    text_model_name: str = field(default="mistralai/Mistral-Nemo-Instruct-2407") # input "chatgpt" to use ChatGPT -- useful for generating training data
    image_model_name: str = field(default="stabilityai/stable-diffusion-3-medium-diffusers")


@dataclass
class DataArguments:
    eval_data_path: str # ../eval_prep/
    output_data_path: str
    dataset_name: str
    num_iter: int = 10
    starting_point: int = 0
    ending_point: int = -1
    batch_size: int = 40
    scoring_column: str = ""

@dataclass
class GenerativeOptimizationArguments:
    num_paraphrases: int = 4
    openai_api_region: str = US_CODE # add your own regions
    optimization_mode: str = "iterative" # one of the following: "iterative", "iterative_for_annotation", "instant", "scoring"
    metrics: str = "vqa-tifa" # can add vqa-tifa-dcs, order or caps does not matter
    gpu_runs: int = 3
    use_distributed: bool = True

@dataclass
class GenerationArguments:
    # generation kwargs -- generally we do not need to modify
    do_sample: bool = False
    num_beams: int = 4
    max_new_tokens: int = 8192
    repetition_penalty: float = 1.2