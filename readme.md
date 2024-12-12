# Fast Prompt Alignment (FPA)

Paper available here: [Arxiv link](https://arxiv.org/abs/2412.08639).

**Authors:**
- **[Khalil Mrini](https://khalilmrini.github.io/)**
- **Hanlin Lu**  
- **Linjie Yang**  
- **Weilin Huang**
- **Heng Wang**

## Table of Contents
- [Introduction](#introduction)
- [File Structure](#file-structure)
- [Data Directories](#data-directories)
  - [`data/`](#data)
  - [`data_prepped/`](#data_prepped)
  - [`data_utils/`](#data_utils)
- [Main Python Script Folders](#main-python-script-folders)
  - [`inference/`](#inference)
  - [`generation/`](#generation)
  - [`in_context_learning/`](#in_context_learning)
  - [`training/`](#training)
- [Bash Scripts for Experimentation and Utility](#bash-scripts-for-experimentation-and-utility)
  - [`research_exp_scripts/`](#research_exp_scripts)
  - [`util_scripts/`](#util_scripts)

## Introduction

Fast Prompt Alignment (FPA) is a novel framework for optimizing prompts in text-to-image generation, enhancing alignment efficiency between complex textual inputs and generated visuals. Using results from iterative prompt optimization, FPA leverages large language models (LLMs) for a one-pass paraphrasing approach, which enables near-real-time prompt optimization with minimal computational demands. By using optimized prompts from real user-generated prompts destined for Midjourney for fine-tuning or in-context learning, FPA significantly improves alignment without sacrificing quality, as validated by both automated metrics and expert human evaluations. This repository provides the code and dataset needed to implement and further develop the FPA method, offering a scalable, efficient solution for real-time, high-demand text-to-image applications.

## File Structure

```plaintext
Fast Prompt Alignment
├── __init__.py
├── data/
│   ├── coco_captions.txt
│   ├── original_descriptions_42k.txt
│   └── partiprompt.txt
├── data_prepped/
│   ├── coco_captions.csv
│   ├── coco_captions_best_paraphrase.csv
│   ├── coco_captions_eval_q_a.json
│   ├── coco_captions_noun_chunks.csv
│   ├── original_descriptions_42k.csv
│   ├── original_descriptions_42k_best_paraphrase.csv
│   ├── original_descriptions_42k_eval_q_a.json
│   ├── original_descriptions_42k_noun_chunks.csv
│   ├── partiprompt.csv
│   ├── partiprompt_best_paraphrase.csv
│   ├── partiprompt_eval_q_a.json
│   └── partiprompt_noun_chunks.csv
├── data_utils/
│   ├── __init__.py
│   ├── constants.py
│   ├── data_sampling.py
│   ├── get_noun_chunks.py
│   └── get_visual_questions.py
├── generation/
│   ├── __init__.py
│   └── generation.py
├── in_context_learning/
│   ├── __init__.py
│   └── icl_prompting.py
├── inference/
│   ├── __init__.py
│   ├── arguments.py
│   ├── constants.py
│   ├── dataset.py
│   ├── generator.py
│   ├── inference.py
│   ├── openai_client.py
│   ├── prompt_optimizer.py
│   ├── scorer.py
│   ├── timer_collection.py
│   └── utils.py
├── research_exp_scripts/
│   ├── generation_mistral.sh
│   ├── mistral.sh
│   ├── prepare_annotation_files.sh
│   ├── score_original_test_prompts.sh
│   └── train_data_generation.sh
├── training/
│   ├── __init__.py
│   ├── compile_existing_data.py
│   ├── finetune.py
│   └── format_input_data.py
└── util_scripts/
    ├── constants.sh
    ├── functions.sh
    ├── init.sh
    ├── run.sh
    └── train.sh
```

## Data Directories

**IMPORTANT: We are not able to release the data at this time. The following describes in detail the format of the data files we have used.**

### `data/`
The `data/` directory contains raw textual data that serves as the foundation for generating and evaluating prompts across three distinct datasets:

- **coco_captions.txt**: Contains a set of image captions from the COCO dataset, presented in plain text format. Each line represents a caption, describing an image’s content with short, descriptive sentences. These captions serve as input prompts or targets for paraphrasing and evaluation.

- **original_descriptions_42k.txt**: A collection of 42,000 prompts sampled from Midjourney’s user-generated prompts. Each line is a description crafted by users to generate images, providing insights into creative language patterns and visual expectations. This dataset represents real-world text-to-image prompt usage, offering a diverse source of inputs for alignment and paraphrasing tasks.

- **partiprompt.txt**: Contains prompts from the PartiPrompts dataset, which is a dataset created for evaluating text-to-image models. These prompts are curated to cover a range of visually and semantically rich scenarios that test the model’s ability to interpret and generate complex visual scenes from text. This dataset enables focused experimentation on prompt alignment and model performance.

### `data_prepped/`
The `data_prepped/` directory includes preprocessed versions of each raw dataset file from `data/`, formatted to support specific tasks and analyses. Each dataset (COCO Captions, Original Descriptions 42K, and PartiPrompts) contains six structured files prepared for various experiments:

For each dataset, the files are as follows:

- **[dataset_name].csv**: A CSV file containing the primary captions or prompts and their rephrased counterparts, in an instruction format suited for fine-tuning. For example:
  - `coco_captions.csv`
  - `original_descriptions_42k.csv`
  - `partiprompt.csv`

- **[dataset_name]_best_paraphrase.csv**: This file contains high-quality paraphrases of the original prompts or captions. These paraphrased versions are selected during 2 iterations of the iterative prompt optimization process with the TIFA and VQA scores. Example files:
  - `coco_captions_best_paraphrase.csv`
  - `original_descriptions_42k_best_paraphrase.csv`
  - `partiprompt_best_paraphrase.csv`

- **[dataset_name]_eval_q_a.json**: A JSON file comprising questions and answers generated for each prompt or caption. Each entry includes the prompt, a series of questions designed to test comprehension and alignment, answer choices, and correct answers. This file supports TIFA scoring. Example files:
  - `coco_captions_eval_q_a.json`
  - `original_descriptions_42k_eval_q_a.json`
  - `partiprompt_eval_q_a.json`

- **[dataset_name]_noun_chunks.csv**: This CSV file extracts and organizes noun chunks (key noun phrases) from each prompt or caption. These noun phrases help identify essential visual elements, and are essential for VQA scoring. Example files:
  - `coco_captions_noun_chunks.csv`
  - `original_descriptions_42k_noun_chunks.csv`
  - `partiprompt_noun_chunks.csv`

### `data_utils/`
The `data_utils/` folder contains utility scripts that perform various data processing tasks essential for preparing and analyzing prompt datasets. Each script is designed to handle a specific function:

- **constants.py**: This file defines constants used throughout the data processing scripts. It serves as a centralized location for defining and managing shared variables across scripts.

- **data_sampling.py**: A script for sampling subsets of data from the Midjourney user prompt dataset. This utility enables users to generate representative samples for testing or analysis, which can be especially useful for handling large datasets more efficiently.

- **get_noun_chunks.py**: This script extracts noun chunks (key noun phrases) from each prompt or caption. This is for VQA scoring.

- **get_visual_questions.py**: A script that generates visual questions, answer choices, and correct answers from captions or prompts. This is for TIFA scoring.

## Main Python Script Folders

### `inference/`
The `inference/` directory contains core scripts for the inference pipeline, including configuration, data handling, prompt generation, and evaluation functions. Here’s a description of each file:

- **arguments.py**: Defines and parses command-line arguments for configuring the inference process, allowing users to specify parameters like batch size, number of iterations, and model configurations.

- **constants.py**: Contains constant definitions for shared values across inference modules, such as dataset-specific paths, score column names, and annotation folder structures.

- **dataset.py**: Manages dataset loading and processing, including functions to retrieve prompts, answers, and other data elements from structured input sources.

- **generator.py**: Responsible for generating prompts based on input configurations, leveraging a model pipeline to create text-to-image or image-to-text outputs for inference tasks.

- **inference.py**: The main inference engine that orchestrates data flow through the generation and scoring processes, handling tasks like batching, model interactions, and output collection.

- **openai_client.py**: Manages interactions with the OpenAI API, facilitating text generation and image analysis through external large language model services.

- **prompt_optimizer.py**: Implements functions to refine and optimize prompts for better alignment and improved scoring, using techniques like paraphrasing and iterative feedback.

- **scorer.py**: Calculates alignment scores for generated prompts and images, including functions to handle metrics like CLIP, TIFA, and VQA scores.

- **timer_collection.py**: Contains a timer utility for recording and averaging execution times for various processes, with capabilities to export results to CSV for performance analysis.

- **utils.py**: Provides miscellaneous utility functions for image encoding, GPU management, and system-level operations, including processes for launching background scripts and handling GPU resources.

### `generation/`
The `generation/` folder includes scripts that handle prompt generation tasks, specifically for fine-tuned LLMs.

- **generation.py**: Contains functions to generate improved prompts on a fine-tuned LLM. It utilizes the OpenAI API, with functions to process and save results for test datasets.

### `in_context_learning/`
The `in_context_learning/` folder consists of code designed for in-context learning with a (larger) LLM.

- **icl_prompting.py**: Implements methods for in-context prompt improvement by loading examples of prompts and their high-quality paraphrases, and applying them to refine user prompts. It supports batch processing of prompts, generating outputs based on pre-defined in-context examples and integrating external API responses for prompt enhancement.

### `training/`
The `training/` folder contains scripts and configurations necessary for compiling, formatting, and fine-tuning LLMs on structured datasets for enhanced text-to-image alignment.

- **compile_existing_data.py**: Aggregates processed datasets by combining and de-duplicating files, ensuring a structured dataset ready for model training or evaluation, with options to remove temporary files after compilation.
  
- **finetune.py**: Facilitates model fine-tuning with parameters for tokenization, LoRA configurations, and specific model arguments, using Hugging Face's Trainer for causal language models. It supports configurable evaluation and saving mechanisms tailored for efficient fine-tuning on GPU.

- **format_input_data.py**: Formats raw input data into various predefined structures compatible with specific models (e.g., Hugging Face, Qwen, Gemma), converting prompts into detailed formats suitable for training and inference with text-to-image alignment objectives.

## Bash Scripts for Experimentation and Utility

This section provides an overview of the bash scripts included in the `research_exp_scripts/` and `util_scripts/` folders. Each script serves specific functions in data preparation, model training, and inference, facilitating streamlined and reproducible experimentation.

### `research_exp_scripts/`
The `research_exp_scripts/` folder contains scripts for conducting various experiments, such as data generation, scoring, and model-specific evaluations.

- **generation_mistral.sh**: Runs the generation pipeline with the Mistral model, generating improved prompts based on specified evaluation data.
- **mistral.sh**: Initiates training for the Mistral model using specified datasets, calling a general training function with the model configuration.
- **score_original_test_prompts.sh**: Scores the original test prompts using the specified model and scoring configuration.
- **train_data_generation.sh**: Runs data generation for training, generating prompts with ChatGPT in an iterative mode for the specified dataset.

### `util_scripts/`
The `util_scripts/` folder includes utility scripts for environment setup, directory management, and script execution, supporting both infrastructure setup and training pipelines.

- **constants.sh**: Defines global environment variables and dataset paths used across different scripts.
- **functions.sh**: Provides helper functions for directory creation and file management.
- **init.sh**: Installs necessary dependencies and sets up environment configurations for NCCL (multi-GPU communication), preparing the system for distributed training.
- **run.sh**: Contains functions to execute and score inference processes, setting up dataset paths and configurations for iterative and scoring modes.
- **train.sh**: Runs the training pipeline for a specified model, combining datasets, formatting inputs, and launching distributed training for fine-tuning with various configurations.