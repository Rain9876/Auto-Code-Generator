import os
import sys

from numpy.core.fromnumeric import size

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from Utils import *
from dataset.Dataset import CSN_Dataset
import collections
from apex import amp
from rouge_score import rouge_scorer, scoring
import numpy as np
import pandas as pd
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
import nltk
from pathlib import Path
from Logic_Eval import logic_evaluate

set_rand_seeds()
device = 1
torch.cuda.set_device(device)
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def reward_control_flow(x):
    return 0


def reward_bleu(x):
    return x


def reward_rogue(x, y):
    scores = []
    for x1, y1 in zip(x, y):
        scores.append(scorer.score(x1, y1))
    return scores


def reward_bleu(x, y):
    scores = []
    for x1, y1 in zip(x, y):
        x1 = x1.split()
        y1 = y1.split()
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(
            [y1], x1, weights=(0.25, 0.25)
        )
        scores.append(BLEUscore)
    return scores


def calculate_bleu(xs, ys):
    BLEUscore = nltk.translate.bleu_score.sentence_bleu(xs, ys)
    return BLEUscore


def Policy_Gradient_Update(model, tokenizer, data):
    batch_size = data["input_ids"].shape[0]
    T = data["input_ids"].shape[1]
    y = data["decoder_input_ids"].to(device, dtype=torch.long)
    lm_labels = y.clone().detach()
    lm_labels[y == tokenizer.pad_token_id] = -100
    ids = data["input_ids"].to(device, dtype=torch.long)
    mask = data["attention_mask"].to(device, dtype=torch.long)

    # print("input ids:", tokenizer.batch_decode(ids, skip_special_tokens=True))
    ## 1. compute rewards
    # logits_processor = LogitsProcessorList(
    #     [MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),]
    # )
    # outputs = model.greedy_search(ids, max_length=50, logits_processor=logits_processor)
    # print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    # print("--")
    # print("--")
    outputs = model.generate(
        ids,
        max_length=50,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
    )
    # print("generated outputs:", outputs.sequences)

    out_string = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    y_string = tokenizer.batch_decode(y, skip_special_tokens=True)
    print("string:", out_string)
    print("y:", y_string)
    # print(len(outputs.scores))
    # print(outputs.scores[0].shape)
    # RS = [x["rougeL"].fmeasure for x in reward_rogue(out_string, y_string)]
    # print("Rogue Score: {}".format(RS))
    # RS = [x for x in reward_bleu(out_string, y_string)]
    RC = logic_evaluate(out_string, y_string)
    # print("Bleu Score: {}".format(RS))
    RS = RC
    print("Logic Score: {}".format(RC))

    # averageRC = np.mean(RC)
    averageRS = np.mean(RS)
    # unbiasedRC = RC  # - averageRC
    unbiasedRS = RS  # - averageRS
    # might have to split these 2 rewards into 2 policy updates.
    total_rewards = torch.tensor(unbiasedRS, dtype=torch.float).to(device)

    ## 2 compute log p(x_t|x_<t), the label doesn't actually matter ? WRONG code
    # outputs2 = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
    # lm_logits = outputs2.logits
    # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # loss = loss_fct(lm_logits, lm_labels)
    # print(lm_logits.shape)
    # print(lm_logits.view(-1, lm_logits.size(-1)).shape)

    # more accurate algorithmicly
    loss = []
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    for i in range(batch_size):
        generated_output = []
        for j in range(len(outputs.scores)):
            generated_output.append(outputs.scores[j][i])
        generated_output = torch.stack(generated_output, dim=0)
        episode_loss = loss_fct(generated_output, outputs.sequences[i][1:])
        loss.append(episode_loss)
    loss = torch.stack(loss, dim=0)
    print(f"loss:{loss}")
    # print(total_rewards)
    # (B,1)
    # summed_loss = torch.sum(loss, dim=0) * 1 / T
    print(total_rewards.shape, loss.shape)
    weighted_loss = torch.dot(total_rewards.squeeze(0), loss.squeeze(0))

    ## 3 Loss
    L = 1 / batch_size * (weighted_loss)

    return L


def training_per_iteration(model, tokenizer, data, optimizer, lr_sch, pg, amp):
    optimizer.zero_grad()

    y = data["decoder_input_ids"].to(device, dtype=torch.long)

    lm_labels = y.clone().detach()
    lm_labels[y == tokenizer.pad_token_id] = -100

    ids = data["input_ids"].to(device, dtype=torch.long)
    mask = data["attention_mask"].to(device, dtype=torch.long)

    outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)

    if pg:
        pg_loss = Policy_Gradient_Update(model, tokenizer, data)
        print(f"NLL{ outputs[0]}, final pg loss:{pg_loss}")

        loss = outputs[0] + pg_loss
    else:
        loss = outputs[0]
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    # loss.backward()
    # print(torch.cuda.memory_summary(device=1, abbreviated=True))
    optimizer.step()

    lr_sch.step()

    return loss


def validate(tokenizer, model, loader, generate=True,generate_method="greedy", interval=1000):

    model.eval()
    val_loss = []
    predictions = []
    actuals = []

    with torch.no_grad():

        for iteration, data in enumerate(loader, 1):

            source_ids, source_mask, y = CSN_Dataset.trim_seq2seq_batch(
                data, tokenizer.pad_token_id
            )

            y = y.to(device, dtype=torch.long)

            lm_labels = y.clone().detach()
            lm_labels[y == tokenizer.pad_token_id] = -100

            ids = source_ids.to(device, dtype=torch.long)
            mask = source_mask.to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)

            loss = outputs[0]
            val_loss.append(loss.item())

            if generate:
                t0 = time.time()
                if generate_method=="greedy":
                    generated_ids = model.generate(
                        input_ids=ids,
                        attention_mask=mask,
                        max_length=300,
                        num_beams=4,
                        early_stopping=True,
                    )
                elif generate_method == "nucleus":
                    generated_ids = model.generate(
                        input_ids=ids,
                        attention_mask=mask,
                        max_length=300,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=True, top_p=0.92
                    )

                preds = [
                    tokenizer.decode(
                        g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    for g in generated_ids
                ]
                target = [
                    tokenizer.decode(
                        t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    for t in y
                ]
                gen_time = (time.time() - t0) / ids.shape[0]

                if iteration % 100 == 0:
                    print(
                        f"[Generation] batch: {iteration}/{len(loader)}, Loss: {sum(val_loss[-100:]) / 100}, Time per gen: {gen_time}s"
                    )

                predictions.extend(preds)
                actuals.extend(target)

            else:
                if iteration % interval == 0:
                    print(
                        f"[Validation] batch: {iteration}/{len(loader)}, Loss: {sum(val_loss[-interval:]) / interval}"
                    )

    return predictions, actuals, val_loss


def save_checkpoint(
    step,
    model,
    opt,
    lr_sch,
    loss,
    min_loss,
    amp,
    pg,
    config,
    best_model=True,
    last_model=True,
    suffix="",
):
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "lr_sch_state": lr_sch.state_dict(),
        "amp": amp.state_dict(),
    }
    folder = str(Path(config.RESUME_PATH).parent)

    if last_model:
        model_name = folder + f"last_model{suffix}.ckpt"
        torch.save(checkpoint, model_name)
        print(f"[Info] The last model on step {step} has been saved.")

    if best_model:
        if loss <= min_loss:
            model_name = folder + f"best_model_loss{suffix}.ckpt"
            torch.save(checkpoint, model_name)
            print("[Info] The best model loss checkpoint file has been updated.")


def load_checkpoint(PATH, model, opt, lr_sch, opt_level):
    checkpoint = torch.load(PATH)

    model, opt = amp.initialize(model, opt, opt_level=opt_level)
    model.load_state_dict(checkpoint["model_state"])
    opt.load_state_dict(checkpoint["optimizer_state"])
    step = checkpoint["step"] + 1
    lr_sch = lr_sch.load_state_dict(checkpoint["lr_sch_state"])

    return model, opt, step, lr_sch


def data_process(config, tokenizer, data_type, data_size):
    # Creating the Training and Validation dataset for further creation of Dataloader
    data_set = CSN_Dataset(
        config.DATA_DIR,
        tokenizer,
        config.MAX_SRC_LEN,
        config.MAX_TGT_LEN,
        data_type,
        data_size,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": config.TRAIN_BATCH_SIZE,
        "shuffle": False,
        "num_workers": 4,
    }

    val_params = {
        "batch_size": config.VALID_BATCH_SIZE,
        "shuffle": False,
        "num_workers": 2,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    if "train" in data_type:
        data_loader = DataLoader(data_set, **train_params)
        print("Training_samples:", len(data_set))
    elif "val" in data_type:
        data_loader = DataLoader(data_set, **val_params)
        print("Val_samples:", len(data_set))
    elif "test" in data_type:
        data_loader = DataLoader(data_set, **val_params)
        print("Test_samples:", len(data_set))

    return data_loader


def produce_predictions(config, model,tokenizer,test_loader,folder, method,file_name):
    prediction, actual, test_loss = validate(
        tokenizer, model, test_loader, generate=True,generate_method=method
    )
    # rouge = calculate_rouge(prediction, actual)
    # bleu = calculate_bleu(prediction, actual)
    # # bleu =
    # # metrics["rouge"].append(rouge)
    # print("Rouge score", rouge)
    # print("Bleu score", bleu)

    final_df = pd.DataFrame({"Generated Code": prediction, "Actual Code": actual})
    folder = str(Path(config.RESUME_PATH).parent)
    final_df.to_csv("./" + folder + file_name)
    return (prediction,)


def fine_tuning():

    param = collections.namedtuple(
        "param",
        [
            "TRAIN_BATCH_SIZE",
            "VALID_BATCH_SIZE",
            "TEST_BATCH_SIZE",
            "TRAIN_EPOCHS",
            "TRAIN_STEPS",
            "TEST_EPOCHS",
            "LEARNING_RATE",
            "MAX_SRC_LEN",
            "MAX_TGT_LEN",
            "DATA_DIR",
            "RESUME_PATH",
            "RESUME",
            "POLICY_GRADIENT",
        ],
    )

    config = param(
        TRAIN_BATCH_SIZE=2,  # input batch size for training_loader
        VALID_BATCH_SIZE=4,  # input batch size for testing
        TEST_BATCH_SIZE=4,  # input batch size for testing
        TRAIN_EPOCHS=100,  # number of epochs to train
        TRAIN_STEPS=5000,
        TEST_EPOCHS=1,
        LEARNING_RATE=1e-5,  # learning rate (default: 0.01)
        MAX_SRC_LEN=512,
        MAX_TGT_LEN=512,
        DATA_DIR="/home/junliw/CodeGeneration/Auto-Code-Generator/data_processing/processed_data/finetuning",

        # RESUME_PATH="/home/junliw/CodeGeneration/Auto-Code-Generator/finetuning/pg/last_model.ckpt",
        RESUME_PATH="/home/junliw/CodeGeneration/Auto-Code-Generator/finetuning/pg_logic/pg_bleubest_model_loss.ckpt",
        RESUME=False,
        POLICY_GRADIENT=True,
    )

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # T5_config = T5Config().from_pretrained("t5-base")
    # model = T5ForConditionalGeneration(config=T5_config)
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.LEARNING_RATE)

    # Defining the LR scheduler with warm up
    lr_sch = LinearWarmupRsqrtDecayLR(optimizer, 1000)

    print("Initiating Finetuning for the T5 model on dataset")

    step = 0
    epoch = 0
    data_portion = 0

    avg_loss = []
    train_loss = []
    val_loss = []
    test_loss = []
    metrics = {}

    train_interval = 100
    val_interval = 100

    opt_level = "O1"
    # Load Saved Model
    if config.RESUME:
        print("Load saved model!")
        print("Path: ", config.RESUME_PATH)
        model, optimizer, step, lr_sch = load_checkpoint(
            config.RESUME_PATH, model, optimizer, lr_sch, opt_level
        )
        step = 5000
        print("step: ", step)
    else:
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model.to(device)

    training_loader = data_process(config, tokenizer, "train", 5000)
    val_loader = data_process(config, tokenizer, "val", 2000)
    test_loader = data_process(config, tokenizer, "test", 2000)

    folder = str(Path(config.RESUME_PATH).parent)
    Path(folder).mkdir(parents=True, exist_ok=True)

    prediction, actual = produce_predictions(config,model,tokenizer,test_loader,folder,"greedy","before_train_prediction.csv")

    print("+++++++++++++++++++++ Fine tuning ++++++++++++++++++++++++")

    while step < config.TRAIN_STEPS:

        # epoch == step // len(training_loader)

        for _, data in enumerate(training_loader, 0):

            model.train()

            loss = training_per_iteration(
                model, tokenizer, data, optimizer, lr_sch, config.POLICY_GRADIENT, amp
            )

            avg_loss.append(loss.item())

            step += 1

            if step % train_interval == 0:
                print(
                    f"Steps: {step}, batch: {step}/{config.TRAIN_STEPS}, Loss:  {sum(avg_loss[-train_interval:]) / train_interval}, lr: {lr_sch.get_lr()[0]}"
                )
                train_loss_ = sum(avg_loss) / len(avg_loss)
                train_loss.append(train_loss_)

            if step % val_interval == 0:
                _, _, val_loss_ = validate(tokenizer, model, val_loader, generate=False)
                val_avg_loss = sum(val_loss_) / len(val_loss_)
                print("Average val loss ", val_avg_loss)
                val_loss.append(val_avg_loss)
                save_checkpoint(
                    step,
                    model,
                    optimizer,
                    lr_sch,
                    val_avg_loss,
                    min(val_loss),
                    amp,
                    config.POLICY_GRADIENT,
                    config,
                )  # Early stop for best Loss

    # Testing
    print("final validation...")
    prediction, actual = produce_predictions(config,model,tokenizer,test_loader,folder,"greedy","prediction.csv")
    rouge = calculate_rouge(prediction, actual)
    bleu = calculate_bleu(prediction, actual)
    # bleu =
    # metrics["rouge"].append(rouge)
    print("Rouge score", rouge)
    print("Bleu score", bleu)
    print("Output Files generated for review")

    Info = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "rouge": rouge,
        "num_of_epochs": config.TRAIN_EPOCHS,
        "num_of_steps": step,
    }

    saveInfo(Info)


if __name__ == "__main__":
    fine_tuning()
