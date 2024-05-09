from transformers import RobertaTokenizer, pipeline
from datasets import Dataset
import pandas as pd

# data generation pipeline using gpt2
generator = pipeline('text-generation', model='gpt2', framework='pt')

# initialize Roberta tokeniser
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# example seed phrases fpr generating data, will need to add more later probably
areas = {
    "kitchen": "The kitchen is designed for ",
    "living room": "The living room is a perfect spot for ",
    "toilet": "The toilet, typically a smaller room, is used for ",
    "on suite bathroom": "The on suite bathroom offers privacy and is equipped for ",
    "master bedroom": "The master bedroom, a spacious area, is intended for ",
    "garage": "The garage serves as a ",
    "backyard": "The backyard, an open space, is ideal for "
}

# labels for each area
label_dict = {area: i for i, area in enumerate(areas.keys())}

# function to generate a description for a given house area
def generate_description(area, seed_phrase):
    generated_text = generator(seed_phrase, max_length=50, num_return_sequences=1)
    return generated_text[0]['generated_text'].strip()

# generate the descriptions and labels
data = []
for area, seed_phrase in areas.items():
    for _ in range(100):
        description = generate_description(area, seed_phrase)
        data.append({'text': description, 'label': label_dict[area]})

# create a dataframe and then a Hugging Face dataset?
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# tokenise the data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)
dataset = dataset.map(tokenize, batched=True)

# save the data
dataset.save_to_disk('C:/Users/asuka/Desktop/capstone project/Language-Model/LLM_DATA')
