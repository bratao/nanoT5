import json
from pathlib import Path

import datasets
from datasets import Features, Value, DatasetDict, Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split, Sequence

from transformers import T5TokenizerFast

VOCAB_SIZE = 32768

if VOCAB_SIZE % 64 != 0:
    print("Performance warning : the vocab size should be a multiple of 64!")


df_fr = datasets.load_dataset("TucanoBR/GigaVerbo", streaming=True, split="train").filter(lambda x: x['label'] == 1).rename_column("text", "raw_content").select_columns(['raw_content'])
df_fr = df_fr.shuffle(seed=42, buffer_size=1000).take(1_000_000)

def get_dataset():
    features = Features({"input": Value("string"), "output": Value("string")})

    files_names = ["wikipedia_train_ocr_cleaning.jsonl", "llama_finetune_train_ocr_cleaning.jsonl"]
    #files_names = ["wikipedia_train_ocr_cleaning.jsonl"]

    train_data = {"input": [], "output": []}
    for files_name in files_names:
        with open(Path(__file__).parents[1] / f"datasets/{files_name}", "r", encoding="utf-8") as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]
            if 'conversations' in data[0]:
                for linha in data:
                    # In this model, the 2 message is the input and the 3 message is the output
                    for i in range(1, len(linha['conversations']), 2):
                        input_example = linha['conversations'][i]["value"]
                        output_example = linha['conversations'][i + 1]["value"]

                        if input_example and output_example:
                            train_data["input"].append(input_example)
                            train_data["output"].append(output_example)
            else:
                train_data["input"] += [d["input"] for d in data]
                train_data["output"] += [d["output"] for d in data]

    size = 500

    # Trucate the test to only size examples and add the remaining to the training set

    test_data = {"input": train_data["input"][:size], "output": train_data["output"][:size]}
    train_data["input"] = train_data["input"][size:]
    train_data["output"] = train_data["output"][size:]

    return DatasetDict(
        {
            "train": Dataset.from_dict(train_data, features=features),
            "test": Dataset.from_dict(test_data, features=features),
        }
    )

df_input = get_dataset()["train"].rename_column("input", "raw_content").select_columns(['raw_content']).to_iterable_dataset()
df_output = get_dataset()["train"].rename_column("output", "raw_content").select_columns(['raw_content']).to_iterable_dataset()

#df = datasets.concatenate_datasets([df_fr, df_input, df_output])
df = datasets.concatenate_datasets([df_fr, df_input, df_output])


def load_dataset_splits(args):
    if args.mode == "pt":

        df_fr = datasets.load_dataset("TucanoBR/GigaVerbo", streaming=True, split="train").filter(
            lambda x: x['label'] == 1).rename_column("text", "raw_content").select_columns(['raw_content'])
        df_fr = df_fr.shuffle(seed=42, buffer_size=1000).take(10_000_000)

        def get_dataset():
            features = Features({"input": Value("string"), "output": Value("string")})

            files_names = ["wikipedia_train_ocr_cleaning.jsonl", "llama_finetune_train_ocr_cleaning.jsonl"]
            # files_names = ["wikipedia_train_ocr_cleaning.jsonl"]

            train_data = {"input": [], "output": []}
            for files_name in files_names:
                with open(Path(__file__).parents[1] / f"datasets/{files_name}", "r", encoding="utf-8") as f:
                    data = f.readlines()
                    data = [json.loads(d) for d in data]
                    if 'conversations' in data[0]:
                        for linha in data:
                            # In this model, the 2 message is the input and the 3 message is the output
                            for i in range(1, len(linha['conversations']), 2):
                                input_example = linha['conversations'][i]["value"]
                                output_example = linha['conversations'][i + 1]["value"]

                                if input_example and output_example:
                                    train_data["input"].append(input_example)
                                    train_data["output"].append(output_example)
                    else:
                        train_data["input"] += [d["input"] for d in data]
                        train_data["output"] += [d["output"] for d in data]

            size = 500

            # Trucate the test to only size examples and add the remaining to the training set

            test_data = {"input": train_data["input"][:size], "output": train_data["output"][:size]}
            train_data["input"] = train_data["input"][size:]
            train_data["output"] = train_data["output"][size:]

            return DatasetDict(
                {
                    "train": Dataset.from_dict(train_data, features=features),
                    "test": Dataset.from_dict(test_data, features=features),
                }
            )
        
        our_dataset = get_dataset()

        df_output = our_dataset["train"].rename_column("output", "raw_content").select_columns(
            ['raw_content']).to_iterable_dataset()
        
        df_test = our_dataset["test"].rename_column("output", "raw_content").select_columns(
            ['raw_content']).to_iterable_dataset()

        # df = datasets.concatenate_datasets([df_fr, df_input, df_output])
        df = datasets.concatenate_datasets([df_fr, df_output])
    

        dataset_splits = {
            "train": df,
            "test": df_test,
        }

    elif args.mode == "ft":
        dataset_splits = datasets.load_dataset(
            args.data.exec_file_path,
            data_dir=args.data.data_dir,
            task_dir=args.data.task_dir,
            max_num_instances_per_task=args.data.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.data.max_num_instances_per_task,
        )
    else:
        raise NotImplementedError

    return dataset_splits


def batch_iterator(dataset, batch_size=1000):
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["raw_content"]

special_tokens_dict = ["<cls>", "<s>", "</s>", "<mask>", "<pad>", "<sep>", "<unk>"]

for i in range(256):
    special_tokens_dict.append("<extra_id_" + str(i) + ">")

# inspired by punct (arXiv:2402.01035v2) but with individual digits
pat_str = r" ?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens_dict, max_token_length=20, min_frequency=2, show_progress=True)
pre_tokenizer = Sequence([Split(pattern=pat_str, behavior="isolated")])
tokenizer.pre_tokenizer = pre_tokenizer

print("Training tokenizer...")

tokenizer.train_from_iterator(batch_iterator(df), trainer)

pretrained_tokenizer = T5TokenizerFast(tokenizer_object=tokenizer, clean_up_tokenization_spaces=False)
pretrained_tokenizer.save_pretrained("tokenizer-t5-gigaverbo")