# Dolly model training with ORT and Lora


import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments

import argparse

from .model_selector import load_tokenizer, load_model
from .data_preprocessing import preprocess_dataset, DataCollatorForCompletionOnlyLM

from .consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY,
    RESPONSE_KEY_NL,
    DEFAULT_TRAINING_DATASET,
)

logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).parent.parent


def train(
    *,
    input_model: str,
    local_output_dir: str,
    dbfs_output_dir: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    lr: float,
    seed: int,
    deepspeed: str,
    gradient_checkpointing: bool,
    local_rank: str,
    bf16: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    test_size: Union[float, int],
    save_total_limit: int,
    warmup_steps: int,
    training_dataset: str = DEFAULT_TRAINING_DATASET,
):
    set_seed(seed)

    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=input_model,
        gradient_checkpointing=gradient_checkpointing,
    )

    # TODO: test the model need to resize the embedding before or now!!
    if apply_lora == True:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)

    # Use the same max lenght that model supports. Fall back to 1024 if setting cannot be found
    # The configuration for the lenght can be stored under different names depending on the model. Here we attempt a few possible names we've encountered

    config = model.config

    max_length = None

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max length: {max_length}")
            break

    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")

    processed_dataset = preprocess_dataset(
        tokenizer=tokenizer,
        max_lenth=max_length,
        seed=seed,
        training_dataset=training_dataset,
    )

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    fp16 = not bf16

    if not dbfs_output_dir:
        logger.warn("Will not save to DBFS")

    training_parameters = dict(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
        warmup_steps=warmup_steps,
    )

    # Apply ort model
    if apply_ort == True:
        training_args = ORTTrainingArguments(**training_parameters)
        trainer = ORTTrainer

    else:
        # use the huggingface trainer
        trainig_args = TrainingArguments(**training_parameters)
        trainer = Trainer

    logging.info("Instantiating Trainer")

    trainer_karg = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    trainer(**trainer_args)
    logging.info("Training")

    trainer.train()

    logger.info(f"Save Model to {local_output}")
    trainer.save_model(output_dir=local_output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        main()
    except Exception:
        logging.exception("main failed")
        raise
