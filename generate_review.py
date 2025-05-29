import pandas as pd
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer & model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("best_model_t5")
model = T5ForConditionalGeneration.from_pretrained("best_model_t5")
model.to(device)
model.eval()

# Data Preparation
def prepare_dataset(data_path):
    df = pd.read_json(data_path, lines=True)
    df = df.rename(columns={'id': 'review_id'})
    ratings_df = df['ratings'].apply(pd.Series)
    extra_columns = [col for col in df.columns if col not in ['text', 'title', 'ratings']]
    processed_df = pd.concat([ratings_df, df[extra_columns], df[['text']]], axis=1)
    processed_df = processed_df.set_index('review_id')
    return processed_df

def clean_text_for_generation(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-z0-9.,!?\'\";:\-\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

test_df = prepare_dataset("hotel_reviews_test.JSON")
test_df['text'] = test_df['text'].apply(clean_text_for_generation)

# Prompt / Build Prompt 
def build_prompt(row, model_type="t5"):
    def describe_score(field, score):
        if score >= 4:
            return f"{field} was excellent ({score})"
        elif score <= 2:
            return f"{field} was poor ({score})"
        else:
            return f"{field} was average ({score})"

    core_fields = ["overall", "cleanliness", "service", "value", "rooms"]
    optional_fields = ["location", "sleep_quality"]
    rating_strs = []
    for field in core_fields + optional_fields:
        score = row.get(field)
        if pd.notna(score):
            rating_strs.append(describe_score(field.replace('_', ' '), score))

    context_parts = []
    if pd.notna(row.get("date_stayed")):
        context_parts.append(f"date stayed: {row['date_stayed']}")
    if pd.notna(row.get("via_mobile")):
        context_parts.append(f"via mobile: {row['via_mobile']}")
    if pd.notna(row.get("num_helpful_votes")) and row['num_helpful_votes'] > 0:
        context_parts.append(f"marked helpful by {row['num_helpful_votes']} users")
    if isinstance(row.get("author"), dict):
        location = row["author"].get("location", "")
        if pd.notna(location) and location != "":
            context_parts.append(f"guest location: {location}")

    prompt = ""
    if rating_strs:
        prompt += "The guest gave the following feedback: " + "; ".join(rating_strs) + "."
    if context_parts:
        prompt += " Additional context: " + ", ".join(context_parts) + "."
    prompt += " Based on this information, write a possible text:"

    return prompt

# Command Line Interface
while True:
    print("\nWelcome! You can generate a review by entering a review_id from the test set.")
    review_id = input("Please enter review_id (or type 'exit' to quit): ").strip()

    if review_id.lower() == 'exit':
        print("Goodbye!")
        break

    if review_id not in test_df.index:
        print(f"Error: review_id '{review_id}' not found in test set.")
        try_again = input("Do you want to try another review_id? (y/n): ").strip().lower()
        if try_again != 'y':
            break
        continue

    # Build Prompt
    row = test_df.loc[review_id]
    prompt = build_prompt(row, model_type='t5')
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    # Generate Review
    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=100)
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nPredicted review text for review_id '{review_id}':\n{output}")
    except Exception as e:
        print(f"Generation failed: {e}")