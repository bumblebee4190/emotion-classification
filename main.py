import os
import time
import json
import re
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import disk_offload
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN in .env file.")

# === Load dataset (use a relative, clean path) ===
df_path = os.path.join("data", "dflyricsoutput.csv")
df = pd.read_csv(df_path)
print("Loaded records:", len(df))

# === Map Valence & Arousal to Emotion Labels ===
def va_map_to_emotion(valence, arousal):
    if 0.7 <= valence <= 1.0 and 0.5 <= arousal <= 1.0:
        return 'love'
    elif 0.5 <= valence < 0.7 and 0.5 <= arousal <= 1.0:
        return 'joy'
    elif 0.3 <= valence < 0.6 and 0.6 <= arousal <= 1.0:
        return 'surprise'
    elif 0.0 <= valence <= 0.3 and 0.7 <= arousal <= 1.0:
        return 'fear'
    elif 0.0 <= valence < 0.4 and 0.6 <= arousal <= 1.0:
        return 'anger'
    elif 0.0 <= valence < 0.4 and 0.0 <= arousal <= 0.6:
        return 'sadness'
    elif 0.6 <= valence <= 1.0 and 0.0 <= arousal <= 0.4:
        return 'calm'
    elif 0.4 <= valence <= 0.6 and 0.4 <= arousal <= 0.6:
        return 'neutral'
    else:
        return 'unknown'

df['emotion_label'] = df.apply(lambda x: va_map_to_emotion(x['valence'], x['arousal']), axis=1)

# === LLM Setup ===
model_id = "openchat/openchat-3.5-1210"
offload_dir = "./offload_folder"
os.makedirs(offload_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
model1 = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder=offload_dir,
    trust_remote_code=True
)

generator = pipeline("text-generation", model=model1, tokenizer=tokenizer)

# === LLM Inference on Dataset ===
target_labels = ["joy", "anger", "sadness", "calm", "surprise", "love", "fear", "unknown"]
MAX_LYRICS_LENGTH = 600

def run_llm_on_dataset(df, output_path, delay=0.0, batch_size=16):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            try:
                results = json.load(f)
                processed_lyrics = {item['lyrics'] for item in results}
                print(f"Resuming from previous run — {len(results)} items loaded.")
            except:
                results = []
                processed_lyrics = set()
    else:
        processed_lyrics = set()

    unprocessed = df[~df['lyrics'].isin(processed_lyrics)]

    for i in range(0, len(unprocessed), batch_size):
        batch = unprocessed.iloc[i:i+batch_size]
        prompts = []

        for _, row in batch.iterrows():
            lyrics = row['lyrics']
            if len(lyrics) > MAX_LYRICS_LENGTH:
                lyrics = lyrics[:MAX_LYRICS_LENGTH] + "..."

            prompt = f"""
You will be given song lyrics, and your job is to classify the emotion expressed.

Lyrics: {lyrics}

Choose from one of the following emotions: {", ".join(target_labels)}.

Please respond in this format:
Emotion: <label>
Confidence: <1-100>
Reason: <short rationale>
"""
            prompts.append(prompt)

        try:
            with torch.no_grad():
                outputs = generator(prompts, max_new_tokens=200, do_sample=False)
        except Exception as e:
            print(f"Error in batch {i}-{i+len(batch)}: {e}")
            continue

        for j, (_, row) in enumerate(batch.iterrows()):
            output = outputs[j][0]["generated_text"].replace(prompts[j], "").strip()
            results.append({
                "title": row.get("song", ""),
                "artist": row.get("artist", ""),
                "lyrics": row["lyrics"],
                "llm_output": output,
                "true_emotion": row.get("emotion_label", "")
            })

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Processed batch {i}–{i+len(batch)} of {len(unprocessed)}")
        if delay > 0:
            time.sleep(delay)

    print(f"Finished. Total saved: {len(results)}")

# Run LLM processing
run_llm_on_dataset(
    df,
    output_path=os.path.join("output", "openchat_results.json")
)

# === Evaluation Function ===
def evaluate_accuracy(results_json):
    with open(results_json, 'r') as f:
        data = json.load(f)

    y_true, y_pred = [], []

    for d in data:
        try:
            pred = re.search(r'Emotion:\s*(\w+)', d['llm_output']).group(1).lower()
            true = d['true_emotion'].lower()
            y_true.append(true)
            y_pred.append(pred)
        except:
            continue

    print(classification_report(y_true, y_pred, labels=target_labels))

# === Parse & Confidence Model ===
def parse_output(entry):
    text = entry["llm_output"]
    try:
        emotion = re.search(r'Emotion:\s*(\w+)', text).group(1)
        confidence = int(re.search(r'Confidence:\s*(\d+)', text).group(1))
        reason = re.search(r'Reason:\s*(.*)', text, re.DOTALL).group(1)
        return emotion, confidence, reason
    except:
        return None, None, None

def build_confidence_model(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    records = []
    for d in data:
        emotion, confidence, reason = parse_output(d)
        if emotion and confidence and reason:
            records.append({
                "lyrics": d["lyrics"],
                "true_emotion": d["true_emotion"],
                "gpt_emotion": emotion,
                "confidence": confidence,
                "reason": reason
            })

    df_conf = pd.DataFrame(records)
    df_conf["correct"] = (df_conf["gpt_emotion"].str.lower() == df_conf["true_emotion"].str.lower()).astype(int)

    tfidf = TfidfVectorizer(max_features=500)
    X = tfidf.fit_transform(df_conf["lyrics"])
    y = df_conf["confidence"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Confidence regression MSE: {mse}")
