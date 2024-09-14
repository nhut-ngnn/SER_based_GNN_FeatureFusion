import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import time
import requests
from utils import read_token, iemocap_dir

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

# Path to the root directory of the IEMOCAP dataset
# iemocap_dir = iemocap_dir()
iemocap_session1_dir = os.path.join(iemocap_dir(), "Session1")

output_dir = "iemocap_features"
os.makedirs(output_dir, exist_ok=True)

def extract_and_save_features(iemocap_dir, processor, model, device, output_dir):
    all_features = []  # List to store all features for consolidated saving
    
    # for session in os.listdir(iemocap_dir):
    #     session_dir = os.path.join(iemocap_dir, session)
    #     if os.path.isdir(session_dir):
    #         for root, _, files in os.walk(session_dir):
    #             for file in files:
    #                 if file.endswith(".wav"):
    #                     audio_path = os.path.join(root, file)
    #                     try:
    #                         # Load the audio file
    #                         waveform, sample_rate = torchaudio.load(audio_path)
                            
    #                         # Preprocess the waveform and extract input values
    #                         input_values = processor(waveform.squeeze(), 
    #                                                  sampling_rate=sample_rate, 
    #                                                  return_tensors="pt").input_values.to(device)
                            
    #                         # Extract features using the pre-trained model
    #                         with torch.no_grad():
    #                             features = model(input_values).last_hidden_state
                            
    #                         # Save individual feature file
    #                         feature_file = os.path.join(output_dir, f"{file}.pt")
    #                         torch.save(features.cpu(), feature_file)
    #                         print(f"Saved features for {file} to {feature_file}")
                            
    #                         # Append features to list for consolidated saving
    #                         all_features.append(features.cpu())
    #                     except Exception as e:
    #                         print(f"Error processing {audio_path}: {e}")
    for root, _, files in os.walk(iemocap_session1_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                try:
                    # Load the audio file
                    waveform, sample_rate = torchaudio.load(audio_path)
                    
                    # Preprocess the waveform and extract input values
                    input_values = processor(waveform.squeeze(), 
                                             sampling_rate=sample_rate, 
                                             return_tensors="pt").input_values.to(device)
                    
                    # Extract features using the pre-trained model
                    with torch.no_grad():
                        features = model(input_values).last_hidden_state
                    
                    # Save individual feature file
                    feature_file = os.path.join(output_dir, f"{file}.pt")
                    torch.save(features.cpu(), feature_file)
                    print(f"Saved features for {file} to {feature_file}")
                    
                    # Append features to list for consolidated saving
                    all_features.append(features.cpu())
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
    if all_features:
        # Save consolidated features list as a single file
        consolidated_file = os.path.join(output_dir, "iemocap_all_features.pt")
        torch.save(all_features, consolidated_file)
        print(f"Saved all features to {consolidated_file}")
    else:
        print("No features were extracted. Please check the input directory and files.")

# Run feature extraction and save features
extract_and_save_features(iemocap_session1_dir, processor, model, device, output_dir)