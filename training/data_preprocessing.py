from .consts import (
    DEFAULT_TRAINING_DATASET,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    RESPONSE_KEY_NL,
)
from functool import partial
import numpy as np
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def load_training_dataset(path_or_dataset: str = DEFAULT_TRAINING_DATASET) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    # only the training dataset from the huggingface
    dataset = load_dataset(path_or_dataset)["train"]
    logger.info(f"Found %d rows", dataset_num_rows)

    # bind the dataset to text
    def _add_text(rec):
        instruction = rec["instruction"]
        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")

        response = rec["response"]
        if not response:
            raise ValueError(f"Excepted an response in: {rec}")

        context = rec.get("context", None)

        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(
                instruction=instruction, response=response, input=context
            )

        if not context:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(
                instruction=instruction, response=response
            )

        return rec

    dataset = dataset.map(_add_text)

    return dataset


def preprocess_batch(
    batch: Dict[str, int], tokenizer: AutoTokenizer, max_length: int
) -> dict:
    return tokenizer(batch["text"], max_length=max_length, trucation=True)


def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    seed=DEFAULT_SEED,
    training_dataset: str = DEFAULT_TRAINING_DATASET,
) -> Dataset:
    """
    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.
    Return:
        Dataset: Huggingface dataset
    """

    dataset = load_training_dataset(training_dataset)

    logger.info("Preprocessing dataset")

    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )

    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we donot have any truncated records, as this would mean the end keyword is missing
    logger.info("Processing dataset has %d rows", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info(
        "Processed dataset has %d rows after filtering for truncated records",
        dataset.num_rows,
    )

    logger.info("shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset
