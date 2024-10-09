import subprocess
import numpy as np
from sklearn.cluster import DBSCAN

def extract_features_with_opensmile(audio_file, config_file, output_file):
    command = f"SMILExtract -C {config_file} -I {audio_file} -O {output_file}"
    subprocess.run(command, shell=True, check=True)

def load_features(output_file):
    features = []
    with open(output_file, 'r') as file:
        for line in file:
            if not line.startswith('@') and not line.startswith('#'):
                features.append([float(x) for x in line.strip().split(',')[1:]])
    return np.array(features)

def cluster_features(features, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    return labels

if __name__ == "__main__":
    audio_file = "path/to/audio.wav"
    config_file = "path/to/opensmile/config.conf"
    output_file = "path/to/output.csv"

    extract_features_with_opensmile(audio_file, config_file, output_file)
    features = load_features(output_file)
    labels = cluster_features(features)

    print("Cluster labels:", labels)