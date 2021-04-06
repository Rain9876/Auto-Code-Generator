import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

T5_PREFIX = "translate NL to Python: "


########################################################################################################################
# Finetuning Datasets
########################################################################################################################


# Removing the padding token for generation

def trim_batch(
        input_ids, pad_token_id, attention_mask=None,
):
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def encode_CSN_file(
        tokenizer,
        data_path,
        max_src_length,
        max_tgt_length,
        start_idx,
        end_idx,
        overwrite_cache=False,
        data_type="",
        prefix=T5_PREFIX):

    cache_path_src = Path(f"{data_path}/{data_type}_src.pt")
    cache_path_tgt = Path(f"{data_path}/{data_type}_tgt.pt")

    if not overwrite_cache and cache_path_tgt.exists() and cache_path_src.exists():

        try:
            cache_src = torch.load(cache_path_src)
            cache_tgt = torch.load(cache_path_tgt)

            assert isinstance(cache_src, list)
            assert isinstance(cache_tgt, list)

            print(f"Load {data_type} cache successful.")

            if end_idx:
                cache_src = cache_src[start_idx:end_idx]
                cache_tgt = cache_tgt[start_idx:end_idx]

            return cache_src, cache_tgt

        except Exception:

            print(f"failed to load from cache, retokenizing at {data_path}")

    else:
            # Process entire database
            data_path = Path(f"{data_path}/{data_type}.csv")

            data = pd.read_csv(data_path, names=["docstring", "code"], header=None)

            print(f"Load {len(data)} {data_type} data.")

            if not prefix:
                data.docstring = prefix + data.docstring

            # data = data.sample(frac=1)

            tokenized_src = tokenizer.batch_encode_plus(
                data["docstring"].values.tolist(),
                max_length=max_src_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            assert tokenized_src.input_ids.shape[1] == max_src_length

            tokenized_tgt = tokenizer.batch_encode_plus(
                data["code"].values.tolist(),
                max_length=max_tgt_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            assert tokenized_tgt.input_ids.shape[1] == max_tgt_length

            encoded_src = [{"input_ids": tokenized_src["input_ids"][i],
                            "attention_mask": tokenized_src["attention_mask"][i]} for i in range(len(data))]

            encoded_tgt = [{"input_ids": tokenized_tgt["input_ids"][i],
                            "attention_mask": tokenized_tgt["attention_mask"][i]} for i in range(len(data))]

            torch.save(encoded_src, cache_path_src.open("wb"))

            torch.save(encoded_tgt, cache_path_tgt.open("wb"))

            if end_idx > len(data):
                end_idx = len(data)

            return encoded_src[start_idx:end_idx], encoded_tgt[start_idx:end_idx]


class CSN_Dataset(Dataset):
    '''
        CSN: CodeSearchNet Dataset

    '''

    def __init__(self, data_path, tokenizer, max_src_length=512, max_tgt_length=512, data_type="train", start_idx=0,
                 end_idx=0):
        self.source, self.target = encode_CSN_file(tokenizer,
                                                   data_path,
                                                   max_src_length,
                                                   max_tgt_length,
                                                   start_idx,
                                                   end_idx,
                                                   overwrite_cache=False,  # Update Once Overwrite Needed
                                                   data_type=data_type)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask, "decoder_input_ids": target_ids}

    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"])
        return source_ids, source_mask, y
