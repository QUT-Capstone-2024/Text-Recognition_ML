from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model_path = 'C:/Users/asuka/Desktop/capstone project/Language-Model/LLM_RoBERTa/trained_model'

# load the tokeniser and model
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

model.eval()

# create dict for labels
label_dict = {
    0: "kitchen",
    1: "living room",
    2: "toilet",
    3: "on suite bathroom",
    4: "master bedroom",
    5: "garage",
    6: "backyard"
}

# use model to predict the label of a given descrption inpout fromt the user
def predict_category(description):
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    return label_dict[predicted_class]

# loop for multiple uses
def interactive_prototype():
    print("Interactive House Area Description Classifier")
    print("Type 'exit' to stop the program.")
    while True:
        description = input("Enter a description: ")
        if description.lower() == 'exit':
            break
        category = predict_category(description)
        print(f"Predicted Category: {category}")

interactive_prototype()
