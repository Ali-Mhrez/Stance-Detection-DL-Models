# Repository of the paper: [Evaluating the Performance of Deep Learning Models on the Task of Stance Detection Towards Fake News](https://journal.homs-univ.edu.sy/index.php/Engineering/article/view/4682), Full text: [pages 69-92](https://journal.homs-univ.edu.sy/index.php/Engineering/issue/view/758/690).

This repository contains the code for a research paper that evaluates several foundational deep learning architectures for stance detection in the context of fake news. The task involves classifying the relationship between a news article's body text and its headline into one of four categories: agree, disagree, discuss, or unrelated. We investigate the performance of various models, including MLP, CNN, BiLSTM, CNN-BiLSTM, BiLSTM-CNN, and an Ensemble of CNN and BiLSTM models.

## Goal and Background
Currently, there's a lack of research on how foundational deep learning models perform on the [AraStance](https://aclanthology.org/2021.nlp4if-1.9/) dataset. This makes it difficult to measure progress and establish a baseline for future studies.

Our goal is to address this gap by investigating the performance of several key deep learning architectures—MLP, CNN, BiLSTM, CNN-BiLSTM, and BiLSTM-CNN—on this dataset. To enhance their performance, we've incorporated transfer learning using [AraVec](https://www.sciencedirect.com/science/article/pii/S1877050917321749) word vectors as model inputs. Additionally, we introduce an Ensemble model (CNN, BiLSTM) to further improve results and provide a comprehensive benchmark for this task.

## Dataset
The [AraStance](https://aclanthology.org/2021.nlp4if-1.9/) dataset includes article bodies, headlines, and a corresponding class label. The label indicates the stance of the article body with respect to the headline. The article body can either Agree (AGR) or Disagree (DSG) with the headline, it can Discuss (DSC) it or be completely Unrelated (UNR).
| Data Source | Data Type | Instances | AGR | DSG | DSC | UNR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| [paper repo](https://github.com/Tariq60/arastance) | News articles | 4,063 | 25.1% | 11.0% | 9.5% | 54.3% |

## Data Preprocessing
The dataset is already divided into: Training, Validation, Testing sets.
<table>
    <thead>
        <tr>
            <th>Step</th>
            <th>Details</th>
            <th>Models</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Concatenation</td>
            <td>Headline + Article body</td>
            <td>ALL</td>
        </tr>
        <tr>
            <td rowspan=2>Cleaning</td>
            <td>remove stop words and punctuation</td>
            <td>MLP</td>
        </tr>
        <tr>
            <td>remove punctuation, diacritics, longation, unicode, and extra spaces</td>
            <td>All but MLP</td>
        </tr>
        <tr>
            <td>Numeric-to-Text</td>
            <td>convert numerics and percentages to texts</td>
            <td>All but MLP</td>
        </tr>
        <tr>
            <td>Normalization</td>
            <td>convert hamza أ إ آ to alif ا, alif maksura ى to ya ي, taa ة to haa ه</td>
            <td>All but MLP</td>
        </tr>
        <tr>
            <td rowspan=2>Vectorizaton</td>
            <td>TF-IDF vectors</td>
            <td>MLP</td>
        </tr>
        <tr>
            <td><a href="https://www.sciencedirect.com/science/article/pii/S1877050917321749">AraVec</a> embeddings</td>
            <td>All but MLP</td>
        </tr>
    </tbody>
</table>

## Models

<table>
      <thead>
            <th>Model</th>
            <th colspan=2>Layers</th>
            <th>Training Hyperparameters</th>
      </thead>
      <tbody>
            <tr>
                  <td>MLP</td>
                  <td colspan=2>Dense (units=4)</td>
                  <td>Batch size=512<br>
                      Learning rate=1e-1<br>
                      epochs=earlystopping(patience=3)
                  </td>
            </tr>
            <tr>
                  <td rowspan=6>CNN</td>
                  <td colspan=2>Embedding (input_dim=30,000, output_dim=300, trainable=False) </td>
                  <td rowspan=6>Batch size=64<br>
                      Learning rate=1e-3<br>
                      epochs=20
                  </td>
            </tr>
            <tr>
                  <td colspan=2>
                        Conv1D (filters=100, kernel_size=2, activation=relu, use_bias=True)<br>
                        Conv1D (filters=100, kernel_size=3, activation=relu, use_bias=True)<br>
                        Conv1D (filters=100, kernel_size=4, activation=relu, use_bias=True)
                  </td>
            </tr>
            <tr>
                  <td colspan=2>
                        GlobalMaxPooling1D<br>
                        GlobalMaxPooling1D<br>
                        GlobalMaxPooling1D
                  </td>
            </tr>
            <tr>
                  <td colspan=2>Concatenation (axis=1)</td>
            </tr>
            <tr>
                  <td colspan=2>Dropout (rate=0.3)</td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
            <tr>
                  <td rowspan=4>BiLSTM</td>
                  <td colspan=2>Embedding (input_dim=20,000, output_dim=300, trainable=False) </td>
                  <td rowspan=4>Batch size=128<br>
                      Learning rate=6e-4<br>
                      epochs=15
                  </td>
            </tr>
            <tr>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=True) ) </td>
            </tr>
            <tr>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=False) ) </td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
            <tr>
                  <td rowspan=5>CNN-BiLSTM</td>
                  <td colspan=2>Embedding (input_dim=30,000, output_dim=300, trainable=False) </td>
                  <td rowspan=5>Batch size=64<br>
                      Learning rate=6e-4<br>
                      epochs=10
                  </td>
            </tr>
            <tr>
                  <td colspan=2>
                        Conv1D (filters=100, kernel_size=2, activation=relu, use_bias=True)<br>
                        Conv1D (filters=100, kernel_size=3, activation=relu, use_bias=True)<br>
                        Conv1D (filters=100, kernel_size=4, activation=relu, use_bias=True)
                  </td>
            </tr>
            <tr>
                  <td colspan=2>Concatenation (axis=1)</td>
            </tr>
            <tr>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=False) ) </td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
            <tr>
                  <td rowspan=7>BiLSTM-CNN</td>
                  <td colspan=2>Embedding (input_dim=20,000, output_dim=300, trainable=False) </td>
                  <td rowspan=7>Batch size=128<br>
                      Learning rate=6e-4<br>
                      epochs=10
                  </td>
            </tr>
            <tr>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=True) ) </td>
            </tr>
            <tr>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=True) ) </td>
            </tr>
            <tr>
                  <td colspan=2>
                        Conv1D (filters=50, kernel_size=2, activation=relu, use_bias=True)<br>
                        Conv1D (filters=50, kernel_size=3, activation=relu, use_bias=True)<br>
                        Conv1D (filters=50, kernel_size=4, activation=relu, use_bias=True)
                  </td>
            </tr>
            <tr>
                  <td colspan=2>
                        GlobalMaxPooling1D<br>
                        GlobalMaxPooling1D<br>
                        GlobalMaxPooling1D
                  </td>
            </tr>
            <tr>
                  <td colspan=2>Concatenation (axis=1)</td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
            <tr>
                  <td rowspan=4>Ensemble (CNN,BiLSTM</td>
                  <td colspan=2>Embedding (input_dim=30,000, output_dim=300, trainable=False) </td>
                  <td rowspan=4>Batch size=128<br>
                      Learning rate=9e-4<br>
                      epochs=12
                  </td>
            </tr>
            <tr>
                  <td>CNN</td>
                  <td>BiLSTM</td>
            </tr>
            <tr>
                  <td colspan=2>Add Logits</td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
      </tbody>
</table>

## Key Results

1. We found that a simple perceptron model using TF-IDF vectors was more effective at stance classification than advanced LSTM and CNN models that utilized transfer learning. This suggests that explicit word cues are highly influential in this particular task.
2. The LSTM models struggled to learn effective contextual representations, likely due to two key characteristics: the long document lengths and the input structure, which combines two distinct text chunks rather than a single, coherent piece of text.
3. The document length and the two-chunk input format appear to be significant challenges for the tested models. Implementing attention techniques in future research could help models overcome these limitations by enabling them to selectively focus on the most important features for accurate classification.

## Requirements

- Python 3.10.12
- NumPy 1.26.4
- TensorFlow 2.17.1
- Keras 3.5.0
- Matplotlib 3.8.0
- Scikit-learn 1.5.2
- Gensim 4.3.3
- PyArabic 0.6.15

## Citation
```bash
@article{amhrez-dl,
author = {Mhrez, ali; Ramadan, Wassim; Abo Saleh, Naser},
title = {Evaluating the Performance of Deep Learning Models on the Task of Stance Detection Towards Fake News},
journal = {Journal of Homs Univeristy},
Series = {Mechanical, Electrical and Information Engineering Sciences Series},
volume = {46},
number = {1},
pages = {69--92},
year = {2024},
url = {https://journal.homs-univ.edu.sy/index.php/Engineering/article/view/4682},
keywords = {stance  detection, fake news, deep learning, natural language processing},
}
```
