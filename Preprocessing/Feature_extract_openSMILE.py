import numpy as np
import os
import sys
import csv
import torch
from utils import execute_opensmile, opensmile_config, iemocap_dir
def csv_reader(add):
    with open(outfilename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        data = np.array(list(reader)).astype(float)
        
    return data[:,2:] 

#### Load data
emotions_used = { 'ang':0, 'hap':1, 'neu':2, 'sad':3 , 'exc':1}
emotions_used_comp = {'Neutral;':2, 'Anger;':0, 'Sadness;':3, 'Happiness;':1}
data_path = iemocap_dir()
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

Label = []
Data = []
fix_len = 120

# openSMILE (Need to be changed)
exe_opensmile = execute_opensmile()
path_config   = opensmile_config()


for sns in sessions:
    emt_label_path = data_path + sns + '/dialog/EmoEvaluation/'
    for file in os.listdir(emt_label_path):
        if file.startswith('Ses'):
            wav_path = data_path + sns + '/sentences/wav/' + file.split('.')[0] + '/'
            ### Reading Emotion labels
            with open(emt_label_path + file, 'r') as f:
                for line in f:
                    if 'Ses' in line:
                        Imp_name = line.split('\t')[1]
                        label = line.split('\t')[2]

                        if not(label.startswith('xxx')) and (label in emotions_used_comp):
                            infilename = wav_path + Imp_name + '.wav'
                            outfilename = "IEMOCAP.csv"
                            opensmile_call = exe_opensmile + " -C " + path_config + " -I " + infilename + " -O " + outfilename
                            os.system(opensmile_call)

                            MFCC = csv_reader(outfilename)
                            label = emotions_used_comp[label]

                            if 'impro' in line:
                                spont_feat = torch.Tensor([1, 0]).view(1, 2).repeat(MFCC.shape[0], 1).detach().cpu().numpy()
                            elif 'script' in line:
                                spont_feat = torch.Tensor([0, 1]).view(1, 2).repeat(MFCC.shape[0], 1).detach().cpu().numpy()

                            MFCC = np.concatenate([MFCC, spont_feat], axis=1)
                            # Splitting MFCC to equal sizes
                            for i in range(MFCC.shape[0] // fix_len):
                                Data.append(MFCC[i * fix_len:(i + 1) * fix_len, :])
                                Label.append(label)

# Save Graph data
np.save('C:/Users/admin/Documents/GNN_SER/SER_based_GNN&FeatureFusion/Dataset/IEMOCAP_data.npy', np.array(Data))
np.save('C:/Users/admin/Documents/GNN_SER/SER_based_GNN&FeatureFusion/Dataset/IEMOCAP_label.npy', np.array(Label))
