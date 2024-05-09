from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

model_path = "C:/Users/SLHan/OneDrive/Desktop/Capstone/Language-Model/Language-Model/LLM_RoBERTa/trained_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval() 

label_dict = {
    0: "kitchen",
    1: "living room",
    2: "toilet",
    3: "on suite bathroom",
    4: "master bedroom",
    5: "garage",
    6: "backyard"
}

# Endpoint to classify description
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    description = data.get('description', '')
    if description:
        inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        category = label_dict[predicted_class]
        return jsonify({'predicted_category': category})
    else:
        return jsonify({'error': 'No description provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
