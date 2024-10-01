from flask import Flask, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Define the API route for summarizing room descriptions
@app.route('/summarize', methods=['POST'])
def summarize_description():
    # Get the JSON data from the request
    data = request.json
    room_description = data.get('description')
    
    if not room_description:
        return jsonify({"error": "No room description provided"}), 400

    # Tokenize input text
    inputs = tokenizer(room_description, max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate summary (condensed description)
    summary_ids = model.generate(
        inputs["input_ids"], 
        num_beams=4, 
        min_length=20, 
        max_length=50, 
        length_penalty=2.0, 
        early_stopping=True
    )
    
    # Decode the output to get the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Return the summary as a JSON response
    return jsonify({"summary": summary})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5100)
