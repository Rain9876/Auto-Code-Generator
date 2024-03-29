from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from Utils import *
from Dataset import PretrainDataset
import collections

set_rand_seeds()
device = set_device()


def training_per_iteration(model, tokenizer, data, optimizer, lr_sch):
    
    optimizer.zero_grad()

    y = data['decoder_input_ids'].to(device, dtype=torch.long)

    lm_labels = y.clone().detach()
    lm_labels[y == tokenizer.pad_token_id] = -100

    ids = data['input_ids'].to(device, dtype=torch.long)
    mask = data['attention_mask'].to(device, dtype=torch.long)

    outputs = model(input_ids=ids, attention_mask=mask, lm_labels=lm_labels)

    loss = outputs[0]

    loss.backward()

    optimizer.step()

    lr_sch.step()

    return loss


def validate(tokenizer, model, loader, generate=True, interval = 1000):
    model.eval()
    val_loss = []
    predictions = []
    actuals = []

    with torch.no_grad():

        for iteration, data in enumerate(loader, 1):

            source_ids, source_mask, y = PretrainDataset.trim_seq2seq_batch(data, tokenizer.pad_token_id)

            y = y.to(device, dtype=torch.long)

            lm_labels = y.clone().detach()
            lm_labels[y == tokenizer.pad_token_id] = -100

            ids = source_ids.to(device, dtype=torch.long)
            mask = source_mask.to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, lm_labels=lm_labels)

            loss = outputs[0]
            val_loss.append(loss.item())

            if generate:
                t0 = time.time()
                generated_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=512,
                    num_beams=2,
                    # repetition_penalty=1.0,
                    # length_penalty=1.0,
                    early_stopping=True
                )

                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                         generated_ids]
                target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
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


def save_checkpoint(step, model, opt, lr_sch, loss, min_loss, best_model=True, last_model=True, suffix=""):
    checkpoint = {'step': step, 'model_state': model.state_dict(), "optimizer_state": opt.state_dict(),
                  "lr_sch_state": lr_sch.state_dict()}
    if last_model:
        model_name = f'last_model{suffix}.ckpt'
        torch.save(checkpoint, model_name)
        print(f'[Info] The last model on step {step} has been saved.')

    if best_model:
        if loss <= min_loss:
            model_name = f'best_model_loss{suffix}.ckpt'
            torch.save(checkpoint, model_name)
            print('[Info] The best model loss checkpoint file has been updated.')


def load_checkpoint(PATH, model, opt, lr_sch):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state'])
    opt.load_state_dict(checkpoint['optimizer_state'])
    step = checkpoint['step']+1
    lr_sch = lr_sch.load_state_dict(checkpoint['lr_sch_state'])
    return model, opt, step, lr_sch


def data_process(config, tokenizer, data_type, data_size):
    # Creating the Training and Validation dataset for further creation of Dataloader
    data_set = PretrainDataset(config.DATA_DIR, tokenizer, config.MAX_SRC_LEN, config.MAX_TGT_LEN, data_type, data_size)

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
    else:
        data_loader = DataLoader(data_set, **val_params)
        print("Val_samples:", len(data_set))

    return data_loader


def load_training_data_portion(current_portion):

        print("Load Data Portion: ", current_portion % 2)

        if current_portion % 2 == 0:
            training_loader = data_process(config, tokenizer, "train0", 250000)
        else:
            training_loader = data_process(config, tokenizer, "train1", 250000)

        return training_loader




def pretraining():
    
    param = collections.namedtuple('param', ["TRAIN_BATCH_SIZE", "VALID_BATCH_SIZE", "TEST_BATCH_SIZE", "TRAIN_EPOCHS", "TRAIN_STEPS",
                                             "TEST_EPOCHS", "LEARNING_RATE", "MAX_SRC_LEN", "MAX_TGT_LEN", "DATA_DIR", "RESUME_PATH"
                                             "RESUME"])

    config = param(
        TRAIN_BATCH_SIZE=8,  # input batch size for training
        VALID_BATCH_SIZE=4,  # input batch size for testing
        TEST_BATCH_SIZE=4,  # input batch size for testing
        TRAIN_EPOCHS=100,  # number of epochs to train
        TRAIN_STEPS = 32000,
        TEST_EPOCHS=1,
        LEARNING_RATE=1e-5,  # learning rate (default: 0.01)
        MAX_SRC_LEN=512,
        MAX_TGT_LEN=512,
        DATA_DIR="/home/song/Desktop/IndividualProject/Pretraining/data/ontol",
        RESUME_PATH = "/home/song/Desktop/pretrain/last_model.ckpt",
        RESUME=True
    )

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    T5_config = T5Config().from_pretrained("t5-base")
    model = T5ForConditionalGeneration(config=T5_config)
    # model = T5ForConditionalGeneration.from_pretrained("t5-base")

    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.LEARNING_RATE)
    
    # Defining the LR scheduler with warm up
    lr_sch = LinearWarmupRsqrtDecayLR(optimizer, 10000)
    


    print('Initiating pretraining for the T5 model on dataset')

    step = 0
    epoch = 0
    data_portion = 0

    avg_loss = []
    train_loss = []
    val_loss = []
    test_loss = []

    metrics = {}

    train_interval = 2000
    val_interval = 1000


    # Load Saved Pretrained Model
    if config.RESUME:
        print("Load pretrain model!")
        print("Path: ", config.RESUME_PATH)
        model, optimizer, step, lr_sch = load_checkpoint(config.RESUME_PATH, model, optimizer, lr_sch)
        print("step: ", step)

    val_loader = data_process(config, tokenizer, "val", 20000)



    print("+++++++++++++++++++++++++ Pretraining  ++++++++++++++++++++++++++")

    while step < config.TRAIN_STEPS:

        # Load portion of processed data
        training_loader = load_training_data_portion(data_portion)
        data_portion  += 1
        epoch = data_portion // 2


        for _, data in enumerate(training_loader, 0):

            model.train()

            loss = training_per_iteration(model, tokenizer, data, optimizer, lr_sch)

            avg_loss.append(loss.item())

            step += 1

            # Print training interval
            if step % train_interval == 0:
                print(
                    f'Step: {step}, batch: {step}/{config.TRAIN_STEPS}, Loss:  {sum(avg_loss[-train_interval:]) / train_interval}, lr: {lr_sch.get_lr()[0]}')
                train_loss_ = sum(avg_loss) / len(avg_loss)
                train_loss.append(train_loss_)

            # Validation
            if step % val_interval == 0:
                _, _, val_loss_ = validate(tokenizer, model, val_loader, generate=False)
                avg_val_loss = sum(val_loss_) / len(val_loss_)
                print("Average val loss ", avg_val_loss)
                save_checkpoint(step, model, optimizer, lr_sch, avg_val_loss, min(val_loss))  # Early stop for best Loss
                val_loss.append(avg_val_loss)



    # Generate Code
    prediction, actual, _ = validate(tokenizer, model, val_loader_generate, generate=True)
    rouge = calculate_rouge(prediction, actual)
    metrics["rouge"].append(rouge)
    print("Rouge score", rouge)
    final_df = pd.DataFrame({'Generated Code': prediction, 'Actual Code': actual})
    final_df.to_csv('./predictions.csv')
    print('Output Files generated for review')
    

    Info = {
        "train_loss": train_loss, 
        "val_loss": val_loss,  
        "test_loss": test_loss,
        "rouge": rouge, 
        "num_of_epochs": config.TRAIN_EPOCHS,
        "num_of_steps": steps
    }

    saveInfo(Info)



if __name__ == '__main__':
    pretraining()
