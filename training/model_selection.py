from transformers import AutoTokenizer, AutoModelForCausalLM
from .consts import END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL, DEFAULT_INPUT_MODEL
from typing import Union, Tuple, Any, Dict, List

def load_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
    )
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    *,
    gradient_checkpointing=gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:

    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True
    )

    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    *,
    gradient_checkpointing: bool = False,
) -> Tuple[AutoModelForCausalLM]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing
    )

    return model, tokenizer

