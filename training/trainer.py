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

from .model_selector import load_tokenizer, load_model, get_model_tokenizer
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


# define str2bool function to handle different input scenario
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        RuntimeError("Boolean value expected")


parser = argparse.ArgumentParser(description="Finetune Dolly Model")

parser.add_argument("--input-model", type=str, help="Input model to fine tune")

parser.add_argument(
    "--local-output-dir", type=str, help="directly local path", required=True
)

# This optional, no need the databricks file system
parser.add_argument(
    "--dbfs-output-dir", type=str, help="sync data to this path on DBFS"
)

parser.add_argument(
    "--epochs", type=int, default=3, help="Number of epochs to train for."
)

parser.add_argument("--apply-ort", type=str2bool, default=False, help="Use the ort")

parser.add_argument("--apply-lora", type=str2bool, default=False, help="Use Lora")

parser.add_argument(
    "--per-device-train-batch-size",
    type=int,
    default=8,
    help="Batch size to use for training.",
)

parser.add_argument(
    "--per-device-eval-batch-size",
    type=int,
    default=8,
    help="Batch size to use for evaluation.",
)

parser.add_argument(
    "--test-size",
    type=int,
    default=1000,
    help="Number of test records for evaluation, or ratio of test records",
)

parser.add_argument(
    "--warmup-steps",
    type=int,
    default=None,
    help="Number of steps to warm up to learning rate",
)

parser.add_argument("--logging-steps", type=int, default=10, help="how often to log")

parser.add_argument(
    "--eval-steps",
    type=int,
    default=50,
    help="How often to run evaluation on test records",
)

parser.add_argument(
    "--save-steps", type=int, default=400, help="How to checkpoint the model"
)

parser.add_argument(
    "--save-total-limit",
    type=int,
    default=10,
    help="Maximum number of checkpoints to keep on disk",
)

parser.add_argument("--lr", type=float, default=1e-5, help="Seed to use for training.")

parser.add_argument(
    "--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training."
)

parser.add_argument(
    "--deepspeed", type=str, default=None, help="Path to deepspeed config file"
)

parser.add_argument(
    "--training-dataset",
    type=str,
    default=DEFAULT_TRAINING_DATASET,
    help="Path to dataset for training",
)

parser.add_argument(
    "--gradient-checkpointing",
    type=str,
    default=True,
    help="Use gradient checkpointing?",
)

parser.add_argument(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identity which instance this process is when performing multi-GPU training",
)


parser.add_argument(
    "--bf16", type=bool, default=None, help="Whether to use bf16 (Preferred on A100's)"
)


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
    apply_ort: bool,
    apply_lora: bool,
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
        max_length=max_length,
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

    logging.info(f"the training parameters for Dolly {training_parameters}")

    # Apply ort model
    if apply_ort == True:
        training_args = ORTTrainingArguments(**training_parameters)

    else:
        # use the huggingface trainer
        training_args = TrainingArguments(**training_parameters)

    logging.info("Instantiating Trainer")

    trainer_args = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    logging.info(f"The trainer arguments {trainer_args}")

    if apply_ort == True:
        trainer = ORTTrainer(**trainer_args)
    else:
        trainer = Trainer(**trainer_args)
    logging.info("Training")

    trainer.train()

    logger.info(f"Save Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    logger.info("Done.")


def main(raw_args=None):
    args = parser.parse_args(raw_args)

    print(args.__dict__)

    training_args_dict = {
        "input_model": args.input_model,
        "local_output_dir": args.local_output_dir,
        "dbfs_output_dir": args.dbfs_output_dir,
        "apply_ort": args.apply_ort,
        "apply_lora": args.apply_lora,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "deepspeed": args.deepspeed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "local_rank": args.local_rank,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "test_size": args.test_size,
        "save_total_limit": args.save_total_limit,
        "warmup_steps": args.warmup_steps,
        "training_dataset": args.training_dataset,
    }

    train(**training_args_dict)


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
