import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import time
import requests
from utils import read_token

HUGGINGFACE_TOKEN = read_token()

# Load the pre-trained model and processor with authentication
model_name = "facebook/wav2vec2-base-960h"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

def load_model_with_retry(model_name, headers, retries=5, backoff_factor=1.0):
    for i in range(retries):
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
            model = Wav2Vec2Model.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
            return processor, model
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"Rate limit exceeded. Retrying in {backoff_factor * (2 ** i)} seconds...")
                time.sleep(backoff_factor * (2 ** i))
            else:
                raise e
    raise Exception("Failed to load model after several retries")

processor, model = load_model_with_retry(model_name, headers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

audio_path = "C:/Users/admin/Documents/GNN_SER/Data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"

waveform, sample_rate = torchaudio.load(audio_path)
input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)

# Extract features using the pre-trained model
with torch.no_grad():
    features = model(input_values).last_hidden_state
print(features)