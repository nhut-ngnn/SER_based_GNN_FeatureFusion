# Speed Emotion Recognition based Fusion Feature and Graph Neural Network </h1>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-04.09.2024-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
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
- [ ] Feature extract with wav2vec 2.0 (on IEMOCAP Dataset)

## Experiment
### Baseline
1. Feature extract
2. Graph Construct
   - Using graph cycle matrix, with topK algorithm =2
3. Graph Convolution Neural Network
   - 2 layers.
   - Numbers of node: 120
   - Numbers of dimenson: 64
4. Pooling Layer: Sumpooling (ablation: Meanpooling, Maxpooling)
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
    <td>100 (early stopping valid_loss = 5)</td>
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
With IEMOCAP, after 5 processing, I have the result follow table:
<table>
  <tr>
    <th>Times</th>
    <th>Weighted Accuracy</th>
    <th>Unweighted Accuracy</th>
  </tr>
  <tr>
    <td>1</td>
    <td>63.24</td>
    <td>63.00</td>
  </tr>
  <tr>
    <td>2</td>
    <td>61.88</td>
    <td>61.59</td>
  </tr>
    <tr>
    <td>3</td>
    <td>61.05</td>
    <td>60.97</td>
  </tr>
    <tr>
    <td>4</td>
    <td></td>
    <td></td>
  </tr>
    <tr>
    <td>5</td>
    <td></td>
    <td></td>
  </tr>
</table>

### Comparison
<table>
   <tr>
      <th>Architecture</th>
      <th>Parameters (MB)</th>
      <th>WA</th>
      <th>UA</th>
   </tr>

   <tr>
      <th>Graph-LSTM (2 layers)</th>
      <th>0.361</th>
      <th>55.93</th>
      <th>63.81</th>
   </tr>

   **<tr>
      <th>Graph-LSTM (3 layers)</th>
      <th>0.409</th>
      <th>59.16</th> 
      <th>68.15</th>
   </tr>**

   <tr>
      <th>Graph-LSTM (4 layers)</th>
      <th>0.591</th>
      <th>58.62</th>
      <th>67.82</th> 

   </tr>
</table>


## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

Email: <link>minhnhut.ngnn@gmail.com </link>

GitHub: <link>https://github.com/nhut-ngnn</link>
