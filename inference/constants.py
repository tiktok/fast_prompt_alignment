US_CODE = "us" ## ADD YOUR OWN COUNTRY CODES HERE

TXT_ONLY = "text"
MULTIMOD = "multimodal"

GPT4O_MODEL = "gpt-4o-2024-05-13"

GPT4O_KEYS = {
    US_CODE: [
        (GPT4O_MODEL, ""), ## ADD YOUR OWN API KEY HERE
    ]
}

MISTRAL_MODEL = "mistral-large"

MISTRAL_KEYS = {
    US_CODE: [
        (MISTRAL_MODEL, "") ## ADD YOUR OWN API KEY HERE
    ]
}

MODEL_AND_API_KEY_BY_TYPE_AND_REGION = {
    "mistral": MISTRAL_KEYS,
    "gpt4o": GPT4O_KEYS,
    TXT_ONLY: {
        US_CODE: GPT4O_KEYS[US_CODE] + [
            ("gpt-4-32k-0613", ""),
            ("gpt-4-0125-preview", ""), ## ADD YOUR OWN API KEYS HERE
        ]
    },
    MULTIMOD: {
        US_CODE: GPT4O_KEYS[US_CODE] + [
            ("gptv", ""), ## ADD YOUR OWN API KEYS HERE
        ]
    }
}

AZURE_ENDPOINTS = {
    TXT_ONLY: {
        US_CODE: "", ## ADD YOUR OWN API ENDPOINT HERE
    },
    MULTIMOD: {
        US_CODE: "", ## ADD YOUR OWN API ENDPOINT HERE
    }
}

AZURE_API_VERSION = "2023-07-01-preview"

HUGGING_FACE_ACCESS_TOKEN = "" ## ADD YOUR OWN HUGGINGFACE ACCESS TOKEN HERE

AVAILABLE_SCORES = ["dcs", "tifa", "vqa"]

SCORE_SUM_COLUMN = "sum"

ITERATIVE_DICT_TEMPLATE = "best_{}_at_each_iter"

ANNOTATION_SUB_FOLDER = "annotations"
PROMPT_COLUMN = "prompt_used_for_generation"
ORIGINAL_COLUMN = "original_prompt"
IMAGE_COLUMN = "image_path"
ANNOTATION_COLUMNS = [PROMPT_COLUMN, ORIGINAL_COLUMN, IMAGE_COLUMN]