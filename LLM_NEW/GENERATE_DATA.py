import openai
import json

# Set your OpenAI API key
openai.api_key = 'sk-proj-qlMvSdPvko8z8sqMEjVHT3BlbkFJU7JpZzHtDhIy2sBcFxy3'

# Categories
categories = ['Bathroom', 'Bedroom', 'Dining', 'Kitchen', 'Livingroom']

# Function to generate sentences using GPT-4
def generate_sentences(prompts, num_sentences=100):
    sentences = []
    for prompt in prompts:
        for _ in range(num_sentences // len(prompts)):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                n=1,
                temperature=0.7
            )
            sentences.append(response.choices[0].message['content'].strip())
    return sentences

# Generate sentences for each category
data = {}
for category in categories:
    prompts = [
        f"Generate a sentence describing a {category.lower()} in a house:",
        f"What does a {category.lower()} look like in a typical house?",
        f"Describe a {category.lower()} found in a home:"
    ]
    data[category] = generate_sentences(prompts, num_sentences=300)  # Adjust number as needed

# Save the generated data to a file
with open('synthetic_room_descriptions.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Generated sentences and saved to 'synthetic_room_descriptions.json'")
