import random
import logging
import sys
import argparse
import os
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def split_label_and_text(line):
    clean = line.decode('utf-8').rstrip('\n')
    (label, text) = clean.split(maxsplit=1)
    
    return (label, text)


def extract_labels_and_texts(lines):
    labels = []
    texts = []
    
    for line in lines:
        label, text = split_label_and_text(line)
        labels.append(label)
        texts.append(text)
        
    return (labels, texts)


def make_number(label):
    if label == "__label__positive":
        return 1
    else:
        return 0
    
    
def load_labels_and_texts(filename):
    lines = []

    with open(filename, 'rb') as f:
        lines = f.readlines()
    
        labels, texts = extract_labels_and_texts(lines)
        
    return (labels, texts)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--valid_dir", type=str, default=os.environ["SM_CHANNEL_VALID"])

    args, _ = parser.parse_known_args()
    
    return args


def setup_logger():
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(message)s",
    )
    
    return logger


def main():
    args = parse_args()
    logger = setup_logger()
    
    train_source = args.training_dir + "/synthetic.train.txt"
    val_source = args.valid_dir + "/synthetic.validation.txt"
    
    train_labels, train_texts = load_labels_and_texts(train_source)
    val_labels, val_texts = load_labels_and_texts(val_source)
    
    logger.info("\n------- 01 --------\n")
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    train_labels_processed = list(map(make_number, train_labels))
    val_labels_processed = list(map(make_number, val_labels))
    
    train_dataset = CustomDataset(train_encodings, train_labels_processed)
    val_dataset = CustomDataset(val_encodings, val_labels_processed)
    
    logger.info("\n------- 02 --------\n")
    
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=3,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
    )

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    logger.info("\n------- 03 --------\n")

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset             
    )
    
    print('DONE')
    
    trainer.train()
    
    logger.info("\n------- 04 --------\n")

    eval_result = trainer.evaluate(eval_dataset=val_dataset)


    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")


            trainer.save_model(args.model_dir)
    
    logger.info("\n------- 05 --------\n")
    

if __name__ == "__main__":
    main()