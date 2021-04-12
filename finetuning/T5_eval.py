from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from Utils import *
from dataset.Dataset import CSN_Dataset
import keyword
import collections
#from apex import amp
from Evaluate import RL_of_loss

set_rand_seeds()
device = set_device()


def training_per_iteration(model, tokenizer, data, optimizer, lr_sch, amp):
    model.train()

    optimizer.zero_grad()

    y = data['decoder_input_ids'].to(device)

    lm_labels = y.clone().detach()
    lm_labels[y == tokenizer.pad_token_id] = -100

    ids = data['input_ids'].to(device)
    mask = data['attention_mask'].to(device)

    outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)

    loss = outputs[0]

    source_ids, source_mask, _ = CSN_Dataset.trim_seq2seq_batch(data, tokenizer.pad_token_id)
    ids = source_ids.to(device)
    mask = source_mask.to(device)

    # print(ids)

    loss = RL_of_loss(model, ids, y, mask, loss, tokenizer)

    #with amp.scale_loss(loss, optimizer) as scaled_loss:
    #    scaled_loss.backward()

    loss.backward()

    optimizer.step()

    lr_sch.step()

    return loss


def validate(tokenizer, model, loader, generate=True, interval=500):
    model.eval()
    val_loss = []
    predictions = []
    actuals = []

    with torch.no_grad():

        for iteration, data in enumerate(loader, 1):

            source_ids, source_mask, y = CSN_Dataset.trim_seq2seq_batch(data, tokenizer.pad_token_id)

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
                generated_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    num_beams=2,
                    early_stopping=True
                )

                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                target = tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                gen_time = (time.time() - t0) / ids.shape[0]

                if iteration % 100 == 0:
                    print(
                        f'[Generation] batch: {iteration}/{len(loader)}, Loss: {sum(val_loss[-100:]) / 100}, Time per gen: {gen_time}s')

                predictions.extend(preds)
                actuals.extend(target)

            else:
                if iteration % interval == 0:
                    print(
                        f'[Validation] batch: {iteration}/{len(loader)}, Loss: {sum(val_loss[-interval:]) / interval}')

    return predictions, actuals, val_loss


def save_checkpoint(step, model, opt, lr_sch, loss, min_loss, amp, data_portion, save_path, best_model=True, last_model=True, suffix=""):
    checkpoint = {'step': step, 'model_state': model.state_dict(), "optimizer_state": opt.state_dict(),
            "lr_sch_state": lr_sch.state_dict(), "amp": amp, "data_portion": data_portion}
    if last_model:
        model_name = f'{save_path}last_model{suffix}.ckpt'
        torch.save(checkpoint, model_name)
        print(f'[Info] The last model on step {step} has been saved.')

    if best_model:
        if loss <= min_loss:
            model_name = f'{save_path}best_model_loss{suffix}.ckpt'
            torch.save(checkpoint, model_name)
            print('[Info] The best model loss checkpoint file has been updated.')


def load_checkpoint(PATH, model, opt, lr_sch):
    checkpoint = torch.load(PATH)

    # model, opt = amp.initialize(model, opt, opt_level=opt_level)
    model.load_state_dict(checkpoint['model_state'])
    opt.load_state_dict(checkpoint['optimizer_state'])
    step = checkpoint['step']
    lr_sch = lr_sch.load_state_dict(checkpoint['lr_sch_state'])
    # data_portion = checkpoint["data_portion"]
    return model, opt, step, lr_sch


def data_process(config, tokenizer, data_type, start_idx, end_idx):
    # Creating the Training and Validation dataset for further creation of Dataloader
    data_set = CSN_Dataset(config.DATA_DIR, tokenizer, config.MAX_SRC_LEN, config.MAX_TGT_LEN, data_type, start_idx,
                           end_idx)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 4
    }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 2
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


def load_training_data_portion(config, tokenizer, current_portion, section=6):
    # Assume 8K per portion and more than 40K for finetuning in total
    # It needs 6 portion as a cycle.
    # It's better to compute how many section in advance, given each volume of portion
    # such as 4K * 11 =~ 40K
    # Load 8K training data per load, 412,179 in total, about 25,761 steps for 16 bz.

    k = current_portion % section

    training_loader = data_process(config, tokenizer, "train", start_idx = 80000 * k, end_idx = 80000 * (k + 1))

    print("Load Data Portion: ", current_portion % section)

    return training_loader


def fine_tuning():
    param = collections.namedtuple('param', ["TRAIN_BATCH_SIZE", "VALID_BATCH_SIZE", "TEST_BATCH_SIZE", "TRAIN_EPOCHS",
                                             "TRAIN_STEPS",
                                             "TEST_EPOCHS", "LEARNING_RATE", "MAX_SRC_LEN", "MAX_TGT_LEN", "DATA_DIR",
                                             "RESUME_PATH","SAVE_PATH",
                                             "RESUME"])

    config = param(
        TRAIN_BATCH_SIZE=16,  # input batch size for training_loader
        VALID_BATCH_SIZE=16,  # input batch size for testing
        TEST_BATCH_SIZE=16,  # input batch size for testing
        TRAIN_EPOCHS=2,  # number of epochs to train
        TRAIN_STEPS=50000,
        TEST_EPOCHS=1,
        LEARNING_RATE=1e-3,  # learning rate (default: 0.01)
        MAX_SRC_LEN=512,
        MAX_TGT_LEN=512,
        DATA_DIR="/home/yurun/Documents/AutoCodeGeneration/data/finetuning/",
        RESUME_PATH="/home/yurun/Documents/AutoCodeGeneration/output/best_model_loss.ckpt",
        SAVE_PATH = "/home/yurun/Documents/AutoCodeGeneration/output/",
        RESUME=True
    )

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.add_tokens(keyword.kwlist)
    tokenizer.add_tokens(["§", "ø"])

    # T5_config = T5Config().from_pretrained("t5-base")
    # model = T5ForConditionalGeneration(config=T5_config)
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.LEARNING_RATE)

    # Defining the LR scheduler with warm up
    lr_sch = LinearWarmupRsqrtDecayLR(optimizer, 5000)

    print('Initiating Finetuning for the T5 model on dataset')

    step = 0
    epoch = 0
    data_portion = 0

    avg_loss = []
    train_loss = []
    val_loss = []
    test_loss = []
    metrics = {}

    train_interval = 200
    val_interval = 1000

    #opt_level = "O1"
    #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    amp = None

    # Load Saved Model
    if config.RESUME:
        print("Load saved model!")
        print("Path: ", config.RESUME_PATH)
        model, optimizer, step, lr_sch = load_checkpoint(config.RESUME_PATH, model, optimizer, lr_sch)
        print("step: ", step)

    # Load 2000 validation data, 23108 in total.
    val_loader = data_process(config, tokenizer, "val", 0, 30000)

    print("+++++++++++++++++++++ Fine tuning ++++++++++++++++++++++++")

    # training_loader = load_training_data_portion(config, tokenizer, data_portion)
    # test_loader = data_process(config, tokenizer, "test", 0, 30000)
    
    #while step < config.TRAIN_STEPS:

        # Load portion of processed data
    #    training_loader = load_training_data_portion(config, tokenizer, data_portion)
        
    #    data_portion += 1
    #    epoch = data_portion // 2

    #    for _, data in enumerate(training_loader, 0):

    #        loss = training_per_iteration(model, tokenizer, data, optimizer, lr_sch, amp)

    #        avg_loss.append(loss.item())

    #        step += 1

    #        if step % train_interval == 0:
    #            print(
    #                f'Steps: {step}, batch: {step}/{config.TRAIN_STEPS}, Loss:  {sum(avg_loss[-train_interval:]) / train_interval}, lr: {lr_sch.get_lr()[0]}')
    #            train_loss_ = sum(avg_loss) / len(avg_loss)
    #            train_loss.append(train_loss_)

    #        if step % val_interval == 0:
    #            _, _, val_loss_ = validate(tokenizer, model, val_loader, generate=False)
    #            val_avg_loss = sum(val_loss_) / len(val_loss_)
    #            print("Average val loss ", val_avg_loss)
    #            val_loss.append(val_avg_loss)
    #            save_checkpoint(step, model, optimizer, lr_sch, val_avg_loss, min(val_loss),
    #                            amp, data_portion, config.SAVE_PATH)  # Early stop for best Loss

    #  Testing

    test_loader = data_process(config, tokenizer, "test", 0, 30000)

    prediction, actual, test_loss = validate(tokenizer, model, test_loader, generate=True)
    rouge = calculate_rouge(prediction, actual)
    print("Rouge score", rouge)
    final_df = pd.DataFrame({'Generated Code': prediction, 'Actual Code': actual})
    final_df.to_csv(f'{config.SAVE_PATH}/predictions.csv')
    print('Output Files generated for review')

    Info = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "rouge": rouge,
        "num_of_epochs": config.TRAIN_EPOCHS,
        "num_of_steps": step
    }

    saveInfo(Info)


# if __name__ == '__main__':
#    fine_tuning()
