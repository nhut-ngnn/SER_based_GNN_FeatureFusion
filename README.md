# Speed Emotion Recognition based Fusion Feature and Graph Neural Network </h1>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-04.09.2024-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>


<p align="center">
<img src="https://img.shields.io/badge/Graph_Neural_Network-white">   
<img src="https://img.shields.io/badge/Feature_Fusion-white">     
<img src="https://img.shields.io/badge/AlexNet-white">
<img src="https://img.shields.io/badge/Sound_Emotion_Recognition-white">
</p>

## Usage 
### Prepare 
```python
pip install -r requirements.txt
```

Install [Opensmile 3.0.2](https://github.com/naxingyu/opensmile).

## Process
- [x] Preprocessing data (on IEMOCAP Dataset) 
- [ ] Feature extract with Librosa library (on IEMOCAP Dataset)
- [x] Feature extract with openSMILE 3.0 (on IEMOCAP Dataset)
- [x] Feature extract with wav2vec 2.0 (on IEMOCAP Dataset)

## Experiment 
### Dataset 
- IEMOCAP with 4 classes: anger, excited, neural, sad.
- Feature extract: using openSMILE 3.0.2 with Interspeech 2009 config.
### Hyperparameter
<table>
  <tr>
    <th>Hyperparameter</th>
    <th>Index</th>
    <th>Hyperparameter</th>
    <th>Index</th>
  </tr>
  <tr>
    <td>Batch size </td>
    <td>128</td>
    <td>K-fold</td>
    <td>5</td>
  </tr>
  <tr>
    <td>Epoch</td>
    <td>100</td>
    <td>Learning rate</td>
    <td>Auto learning rate (min 0.0005)</td>
  </tr>
  <tr>
    <td>GCN layers</td>
    <td>2</td>
    <td>Pooling layer</td>
    <td>Sum Pooling, I also tried with MaxPooling and MeanPooling.</td>
  </tr>
</table>
### Result

## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

Email: <link>minhnhut.ngnn@gmail.com </link>

GitHub: <link>https://github.com/nhut-ngnn</link>
