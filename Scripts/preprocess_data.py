import json
import spacy
from textblob import TextBlob
import os

# Load spaCy's English model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def anonymize_text(text):
    """
    Anonymizes text by replacing PERSON and GPE (Geopolitical Entity)
    entities with a generic tag.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE"]:
            text = text.replace(ent.text, f"[{ent.label_}]")
    return text

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the text and returns a simple tag.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def process_file_to_chat_jsonl(input_file_path, output_file_path):
    """
    Reads a text file, processes each line, and saves it to a JSONL file
    in a chat-based fine-tuning format.
    """
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        raw_text_entries = infile.readlines()

    system_message = "You are NeuroMirror, a helpful AI assistant that reflects the user's past self and helps with self-reflection. Your responses should be grounded, supportive, and in the user's unique tone and style."

    for entry in raw_text_entries:
        if entry.strip() == "":
            continue

        # Anonymize the text
        anonymized_entry = anonymize_text(entry).strip()

        # Get sentiment
        sentiment = analyze_sentiment(anonymized_entry)

        # Manually craft the 'assistant' response.
        # This is the most crucial part where you teach the model to reflect "you".
        if sentiment == "negative":
            assistant_response = f"I'm hearing a lot of frustration here. This feels like a moment to pause and find some grounding. Let's explore why you're feeling this way and what your most resilient self would do."
        elif sentiment == "positive":
            assistant_response = f"There's a really hopeful energy in this thought. It's great to see this kind of momentum. Let's build on this positive feeling and think about what comes next."
        else: # Neutral
            assistant_response = f"This is a very clear and considered thought. It seems like you're processing things logically. What is the next logical step?"

        # Create the chat-based entry
        chat_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": anonymized_entry
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ]
        }
        data.append(chat_entry)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for item in data:
            outfile.write(json.dumps(item) + '\n')
            
    print(f"Processed {len(data)} entries and saved to {output_file_path}")

# --- Example Usage ---
# Create a dummy input file for demonstration
dummy_text_content = """
Today felt like I was running in circles, just endless tasks and no real progress. I'm so tired of feeling stuck.
Got a new project at work and I'm genuinely excited about the challenge. This is exactly what I needed.
I've been thinking a lot about the book I read last week and its central theme. It's complex but fascinating.
"""
with open("raw_data.txt", "w", encoding="utf-8") as f:
    f.write(dummy_text_content)

# Run the processing pipeline
process_file_to_chat_jsonl("raw_data.txt", "neuro_mirror_data.jsonl")
