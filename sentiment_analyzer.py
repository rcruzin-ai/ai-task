# sentiment_analyzer.py

import requests
import pandas as pd

class SentimentAnalyzer:
    
    def __init__(self):
        self.sentiment_mapping = {
            "positive": ["positive", "POS", "LABEL_2"],
            "neutral": ["neutral", "NEU", "LABEL_1"],
            "negative": ["negative", "NEG", "LABEL_0"]
        }

    def standardize_sentiment_label(self, label):
        for key, value in self.sentiment_mapping.items():
            if label.lower() in [x.lower() for x in value]:
                return key
        return label

    def convert_to_df(self, result):
        sentiment_df = pd.DataFrame(result)
        sentiment_df.set_index("label", inplace=True)
        return sentiment_df

    def analyze_text(self, text, model_path):
        api_endpoint = f"https://api-inference.huggingface.co/models/{model_path}"
        headers = {"Content-Type": "application/json"}
        payload = {"inputs": text}
    
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            standardized_result = [{"label": self.standardize_sentiment_label(x["label"]), 
                                    "score": round(x["score"] * 100, 2)} for x in result[0]]
            return standardized_result
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while sending the request: {e}")
            return None
