from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# load the tokenised data
dataset = load_from_disk('C:/Users/asuka/Desktop/capstone project/Language-Model/LLM_DATA')

# split dataset into train and test
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
validation_dataset = split_dataset['test']

# tokeniser
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# load the model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(set(dataset['label'])))

# define the training arguments
training_args = TrainingArguments(
    output_dir='./results',  
    num_train_epochs=3,  
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='steps',
    save_steps=500,
    eval_steps=500,
    logging_dir='./logs',
    learning_rate=5e-5,
)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)

# train the model
trainer.train()

# save the trained model
model.save_pretrained('C:/Users/asuka/Desktop/capstone project/Language-Model/LLM_RoBERTa/trained_model')
tokenizer.save_pretrained('C:/Users/asuka/Desktop/capstone project/Language-Model/LLM_RoBERTa/trained_model')