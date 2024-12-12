from transformers import HfArgumentParser
from accelerate import Accelerator
from inference.openai_client import OpenAIClient
from inference.scorer import Scorer
from inference.generator import Generator
from inference.dataset import Dataset
from inference.prompt_optimizer import PromptOptimizer
from inference.utils import launch_nohup_script_many_times, kill_multiple_processes
from inference.arguments import *

def generate():
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerativeOptimizationArguments, GenerationArguments))
    model_args, data_args, gen_optim_args, gen_kwargs = parser.parse_args_into_dataclasses()

    process_ids = launch_nohup_script_many_times(num_times=gen_optim_args.gpu_runs)

    # Initializing Accelerator
    if gen_optim_args.use_distributed:
        accelerator = Accelerator()
        device_map = {"": accelerator.local_process_index}
    else:
        accelerator = None
        device_map = None

    # Initializing GPT-4o Client
    gpt4o_client = OpenAIClient(region=gen_optim_args.openai_api_region, model_type="gpt4o", error_file_path=data_args.output_data_path)

    # Initializing Generator
    generator = Generator(gen_optim_args.num_paraphrases, model_args.text_model_name, model_args.image_model_name, accelerator, device_map, gen_optim_args.optimization_mode, gpt4o_client, gen_kwargs)

    # Initializing Scorer
    scorer = Scorer(gen_optim_args.metrics, data_args.output_data_path, data_args.dataset_name, gpt4o_client, generator.image_model_pipeline)

    # Initializing Dataset
    dataset = Dataset(data_args.eval_data_path, data_args.dataset_name, data_args.starting_point, data_args.ending_point, data_args.scoring_column)

    # Get Prompt Optimizer to Iterate
    prompt_optimizer = PromptOptimizer(scorer, generator, dataset, accelerator)
    prompt_optimizer.run(data_args.batch_size, data_args.num_iter)

    kill_multiple_processes(process_ids)


if __name__ == "__main__":
    generate()