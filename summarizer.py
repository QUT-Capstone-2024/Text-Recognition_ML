from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Input sentence
sentence = "I renovated the kitchen floor in 2019 to make it look better and more modern. I also cut down the trees in the back yard because it was just blocking way too much sunlight."

# Prepare input for the model
input_text = "summarize: " + sentence
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
summary_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the summary
print(summary)
