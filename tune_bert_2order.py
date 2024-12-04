'''
Tune Bert-based Models for Text Classification using SST-2 from HuggingFace.

Since dataset SST-2 doesn't include ground-truth labels in the test split,
    this script applies the validation split as the testing set,
    and split the train split for training and validating respectively.

This is a modified version to enable ADAHESSIAN.

!!! WARNING: ADAHESSIAN does not work for Bert. Debugging is not yet finished. !!! @xianwen Dec 4, 2024
'''
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler

import argparse
import random
import time
import copy
import os
import pandas as pd

from utils import load_from_json
from model_optim_utils import generate_optimizer
from addition_optim.ada_hessian import AdaHessian

os.environ["PYTORCH_FORCE_FB_MM_ATTENTION"] = "1"

parser = argparse.ArgumentParser(description='HuggingFace Text-Classification Tuning')
parser.add_argument('--json_file', default='./json/sst2_bert_adamw.json',
                    type=str, help='path to the file containing arguments')
args = parser.parse_args()
print(args)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
### Hyperparameters
hyper_args = load_from_json(args.json_file)
checkpoint = hyper_args['pretrained_checkpoint']  # specify pretrained checkpoint from HuggingFace repository
num_epochs = hyper_args["epoch"]
best_acc = 0  # best eval accuracy
best_state = None  # state of the model with highest acc on the val set
best_epoch = 0  # epoch achieving the highest acc on the val set
record_test_acc = 0  # test acc for the model to save

### Set random seed
seed = hyper_args["seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
# ensure deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


### Set random seed
seed = hyper_args["seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# ensure deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


### Data
print("==> Load Data ...")
raw_datasets = load_dataset("glue", "sst2")  # load dataset from GLUE
dataset_args = hyper_args["dataset_params"]
tokenizer = AutoTokenizer.from_pretrained(checkpoint) 

def tokenize(example):
   tokenized_sentence = tokenizer(example['sentence'], padding=True, truncation=True)
   return tokenized_sentence
# tokenize sentences
tokenized_dataset = raw_datasets.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
# remove the columns "sentence" and "idx"
tokenized_dataset = tokenized_dataset.remove_columns(['sentence', 'idx'])
# rename the column "label" to "labels"
tokenized_dataset = tokenized_dataset.rename_column('label','labels')

# wrap dataset
generator = torch.Generator().manual_seed(seed)
tokenized_dataset_train, tokenized_dataset_val = tokenized_dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=seed).values()
train_dataloader=DataLoader(tokenized_dataset_train, 
                            shuffle=dataset_args["shuffle"], batch_size=dataset_args["batch_size"],
                            collate_fn=data_collator,
                            generator=generator,
                            worker_init_fn=lambda worker_id: random.seed(seed + worker_id))
eval_dataloader=DataLoader(tokenized_dataset_val, 
                           shuffle=dataset_args["shuffle"], batch_size=dataset_args["batch_size"],
                           collate_fn=data_collator,
                           generator=generator,
                           worker_init_fn=lambda worker_id: random.seed(seed + worker_id))
test_dataloader=DataLoader(tokenized_dataset['validation'],
                           shuffle=dataset_args["shuffle"], batch_size=dataset_args["test_batch_size"],
                            collate_fn=data_collator,
                            generator=generator,
                            worker_init_fn=lambda worker_id: random.seed(seed + worker_id))


### Model
print("==> Load Model ...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=2,
                                                           output_hidden_states=True)
model.to(device)

# optimizer
optim_params = hyper_args['optimizer_params']
# optimizer = generate_optimizer(hyper_args['optimizer'], hyper_args['optimizer_params'], model)
optimizer = AdaHessian(model.parameters(), 
                       lr = optim_params['lr'],
                       betas = optim_params['betas'],
                       eps = optim_params['eps'],
                       weight_decay = optim_params['weight_decay'],
                       hessian_power = optim_params['hessian_power'])
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=hyper_args['warmup_step'],
    num_training_steps=num_training_steps,
)
criterion = torch.nn.CrossEntropyLoss()


### Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(train_dataloader):
        batch={k:v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs=model(**batch)

        loss=criterion(outputs.logits, batch["labels"])
        loss.backward(create_graph=True)
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        predicted = torch.argmax(outputs.logits, dim=-1)
        total += batch["labels"].size(0)
        correct += predicted.eq(batch["labels"]).sum().item()

    end_time = time.time()
    print('Training: Loss: %.3f | Acc: %.3f (%d/%d)' % (train_loss/len(train_dataloader), 100.*correct/total, correct, total))
        
    # report the loss, acc, and time used per epoch
    return train_loss/len(train_dataloader), 100.*correct/total, end_time-start_time

### validating
def validate(epoch):
    global best_acc, best_state, best_epoch

    model.eval()

    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            batch={k:v.to(device) for k, v in batch.items()}
            outputs=model(**batch)
            loss=criterion(outputs.logits, batch["labels"])

            val_loss += loss.item()
            predicted = torch.argmax(outputs.logits, dim=-1)
            total += batch["labels"].size(0)
            correct += predicted.eq(batch["labels"]).sum().item()
    print('Validating: Loss: %.3f | Acc: %.3f (%d/%d)' % (val_loss/len(eval_dataloader), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        test_loss, test_acc = test(model)  # save the testing information
        state = {
            'net': copy.deepcopy(model.state_dict()),
            'acc': test_acc,
            'loss': test_loss,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, hyper_args['save']+'_epoch{}_acc{:.2f}.pt'.format(epoch, acc))
        best_acc = acc
        best_state = state
        best_epoch = epoch

    return val_loss/len(eval_dataloader), acc

### Testing
def test(model):
    global record_test_acc

    # print('Testing...')
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            batch={k:v.to(device) for k, v in batch.items()}
            outputs=model(**batch)
            loss=criterion(outputs.logits, batch["labels"])
            
            test_loss += loss.item()
            predicted = torch.argmax(outputs.logits, dim=-1)
            total += batch["labels"].size(0)
            correct += predicted.eq(batch["labels"]).sum().item()
    record_test_acc = 100.*correct/total
    print('Testing: Loss: %.3f | Acc: %.3f (%d/%d)' % (test_loss/len(test_dataloader), record_test_acc, correct, total))
    
    # return the value
    return test_loss/len(test_dataloader), record_test_acc


### Training
trace_dicn = {"epoch": [],
              "train_loss": [], "train_acc": [], "train_time": [],
              "val_loss": [], "val_acc": []}
print('==> Training model..')
for epoch in range(num_epochs):
    # training and validating
    train_loss, train_acc, train_time = train(epoch)
    val_loss, val_acc = validate(epoch)  # testing included if a better model is obtained

    # save training information
    trace_dicn['epoch'].append(epoch)
    trace_dicn['train_loss'].append(train_loss)
    trace_dicn['train_acc'].append(train_acc)
    trace_dicn['train_time'].append(train_time)
    trace_dicn['val_loss'].append(val_loss)
    trace_dicn['val_acc'].append(val_acc)

### Saving results
print('==> Saving results..')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(best_state, hyper_args['save']+'_epoch{}_acc{:.2f}.pt'.format(best_epoch, record_test_acc))

trace_df = pd.DataFrame(trace_dicn)
trace_df.to_csv(hyper_args['save']+"_epoch{}best{}.csv".format(num_epochs, best_epoch),
                 index=False)

print("====ALL SET====")
