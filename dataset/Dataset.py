import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from Utils import pretrained_data_preprocess
import gc

T5_PREFIX = "summarize: "


########################################################################################################################
# Finetuning Datasets
########################################################################################################################

def encode_mimic_file(
        tokenizer,
        data_path,
        max_src_length,
        max_tgt_length,
        numOfSamples,
        overwrite_cache=False,
        data_type="",
        prefix=T5_PREFIX):
    cache_path_src = Path(f"{data_path}_{data_type}_src.pt")
    cache_path_tgt = Path(f"{data_path}_{data_type}_tgt.pt")

    if not overwrite_cache and cache_path_src.exists() and cache_path_tgt.exists():
        try:
            examples_src = torch.load(cache_path_src)
            assert isinstance(examples_src, list)
            examples_tgt = torch.load(cache_path_tgt)
            assert isinstance(examples_tgt, list)
            print(f"Load {data_type} dataset successful.")
            if numOfSamples:
                examples_src = examples_src[:numOfSamples]
                examples_tgt = examples_tgt[:numOfSamples]
            return examples_src, examples_tgt

        except Exception:
            print(f"failed to load from {cache_path_src}, retokenizing {data_path}")

    data_path = Path(f"{data_path}/{data_type}.csv")

    data = pd.read_csv(data_path, names=["Source", "Target"], header=None)

    if numOfSamples < len(data):
        data = data.sample(n=numOfSamples, replace=False)
        print(len(data))

    data.Source = data.Source + "</s>"
    data.Target = data.Target + "</s>"

    if not prefix:
        data.Source = prefix + data.Source

    tokenized0 = tokenizer.batch_encode_plus(
        data["Source"].values.tolist(),
        max_length=max_src_length,
        pad_to_max_length=True,
        add_prefix_space=True,
        truncation=True,
        return_tensors="pt"
    )

    assert tokenized0.input_ids.shape[1] == max_src_length

    tokenized1 = tokenizer.batch_encode_plus(
        data["Target"].values.tolist(),
        max_length=max_tgt_length,
        pad_to_max_length=True,
        add_prefix_space=True,
        truncation=True,
        return_tensors="pt"
    )

    assert tokenized1.input_ids.shape[1] == max_tgt_length

    examples_src = [{"input_ids": tokenized0["input_ids"][i],
                     "attention_mask": tokenized0["attention_mask"][i]} for i in range(len(data))]
    examples_tgt = [{"input_ids": tokenized1["input_ids"][i],
                     "attention_mask": tokenized1["attention_mask"][i]} for i in range(len(data))]

    torch.save(examples_src, cache_path_src.open("wb"))
    torch.save(examples_tgt, cache_path_tgt.open("wb"))

    return examples_src, examples_tgt


# Removing the padding token for generation
def trim_batch(
        input_ids, pad_token_id, attention_mask=None,
):
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class MIMICDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_src_length=512, max_tgt_length=128, data_type="train",
                 numOfSamples=10000):
        self.data_path = data_path
        self.tok = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_len = max_tgt_length
        self.numOfSamples = numOfSamples

        self.source, self.target = encode_mimic_file(self.tok, self.data_path, self.max_src_length, self.max_tgt_len,
                                                     self.numOfSamples, overwrite_cache=True, data_type=data_type,
                                                     prefix=T5_PREFIX)

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


########################################################################################################################
# Pretraining Datasets
########################################################################################################################

def encode_pretrain_file(
        tokenizer,
        data_path,
        max_src_length,
        max_tgt_length,
        numOfSamples=None,
        overwrite_cache=False,
        data_type="",
        prefix=None):
    cache_path_src = Path(f"{data_path}/data_{data_type}_src.pt")
    cache_path_tgt = Path(f"{data_path}/data_{data_type}_tgt.pt")

    if not overwrite_cache and cache_path_src.exists() and cache_path_tgt.exists():
        try:
            examples_src = torch.load(cache_path_src)
            assert isinstance(examples_src, list)
            examples_tgt = torch.load(cache_path_tgt)
            assert isinstance(examples_tgt, list)
            print(f"Load {data_type} dataset successful.")
            if numOfSamples:
                examples_src = examples_src[:numOfSamples]
                examples_tgt = examples_tgt[:numOfSamples]
            return examples_src, examples_tgt

        except Exception:
            print(f"failed to load from {cache_path_src}, retokenizing {data_path}")

    print(f"Processing pretrain_{data_type}")

    encode_input, decode_input = pretrained_data_preprocess(tokenizer, f"{data_path}/pretrain_{data_type}.csv",
                                                            numOfSamples)

    print(f"Samples: {len(encode_input)}")

    encode_input_tok = tokenizer.batch_encode_plus(
        encode_input,
        max_length=max_src_length,
        pad_to_max_length=True,
        add_prefix_space=True,
        truncation=True,
        return_tensors="pt"
    )

    assert encode_input_tok.input_ids.shape[1] == max_src_length

    del encode_input
    gc.collect()
    print(f"Input tokenization complete!")

    decode_input_tok = tokenizer.batch_encode_plus(
        decode_input,
        max_length=max_tgt_length,
        pad_to_max_length=True,
        add_prefix_space=True,
        truncation=True,
        return_tensors="pt"
    )

    assert decode_input_tok.input_ids.shape[1] == max_tgt_length

    del decode_input
    gc.collect()
    print(f"Target tokenization complete!")

    examples_src = [{"input_ids": encode_input_tok["input_ids"][i],
                     "attention_mask": encode_input_tok["attention_mask"][i]} for i in
                    range(len(encode_input_tok["input_ids"]))]

    examples_tgt = [{"input_ids": decode_input_tok["input_ids"][i],
                     "attention_mask": decode_input_tok["attention_mask"][i]} for i in
                    range(len(decode_input_tok["input_ids"]))]

    torch.save(examples_src, cache_path_src.open("wb"))
    torch.save(examples_tgt, cache_path_tgt.open("wb"))

    return examples_src, examples_tgt


class PretrainDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_src_len=512, max_tgt_len=128, data_type="train", numOfSamples=None):
        self.data_path = data_path
        self.tok = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.numOfSamples = numOfSamples

        self.source, self.target = encode_pretrain_file(self.tok, self.data_path, self.max_src_len, self.max_tgt_len,
                                                        numOfSamples=self.numOfSamples,
                                                        overwrite_cache=False, data_type=data_type, prefix=None)

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
