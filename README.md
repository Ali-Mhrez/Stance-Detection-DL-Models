# Stance Detection Using Deep Learning Techniques

This repository contains the code and results of experiments on stance detection using various deep learning models.

## Models:

* **Multi-Layer Perceptron (MLP)**
* **Convolutional Neural Network (CNN)**
* **Bidirectional Long Short-Term Memory (BiLSTM)**
* **CNN-BiLSTM**
* **BiLSTM-CNN**
* **Ensemble Model (CNN + BiLSTM)**

## Experiments:

### Dataset:
The experiments were conducted on Google Colab using a AraStance (Alhindi et al., [2021](https://aclanthology.org/2021.nlp4if-1.9/)) dataset.

### Data Preprocessing:

* **Data Splitting:** No need to divide the dataset into training, validation, and testing sets because it is already divided.
* **Concatenation:** concatenate each claim article pair together to create an instance.

#### Specific to MLP
* **Text Cleaning:** remove stop words and punctuation.
* **Vectorization**: using TF-IDF vectors.

#### Specific to the other models
* **Text Cleaning:** remove punctuation, diacritics, longation, unicode, and extra spaces.
* **Numeric-to-Text Conversion**: convert numerics and percentages to texts.
* **Normalization**: convert hamza أ إ آ to alif ا, alif maksura ى to ya ي, taa ة to haa ه.
* **Vectorizaton**: using AraVec (Soliman et al., [2017](https://www.sciencedirect.com/science/article/pii/S1877050917321749)) word embeddings.

### Model Architectures:

* **Multi-Layer Perceptron (MLP):**

      Dense(4)
  
* **Convolutional Neural Network (CNN):**

      [Conv1D(100,2,relu,bias), Conv1D(100,3,relu,bias), Conv1d(100,4,relu,bias)] >
      [  GlobalMaxPooling1D,      GlobalMaxPooling1D,      GlobalMaxPooling1D] >
      Concatenation(axis=1) > Dropout(0.3) > Dense(4)

* **Bidirectional Long Short-Term Memory (BiLSTM):** 

      BiLSTM(32,return_sequences) > BiLSTM(32) > Dense(4)
  
* **CNN-BiLSTM:**

      [Conv1D(100,2,relu,bias), Conv1D(100,3,relu,bias), Conv1d(100,4,relu,bias)] >
      Concatenation(axis=1) > BiLSTM(32) > Dense(4)
  
* **BiLSTM-CNN:** 

      BiLSTM(32,return_sequences) > BiLSTM(32,return_sequences) >
      [Conv1D(50,2,relu,bias), Conv1D(50,3,relu,bias), Conv1d(50,4,relu,bias)] >
      [ GlobalMaxPooling1D,     GlobalMaxPooling1D,     GlobalMaxPooling1D] >
      Concatenation(axis=1) > Dense(4)
  
* **Ensemble Model (CNN + BiLSTM):** 

      (logits from CNN + logits from BiLSTM) / 2

### Evaluation Metrics:

* **Accuracy:** the ratio of the number of correct predictions to the total number of predictions.
* **F1-Score:** the harmonic mean of precision and recall.
* **Macro F1-score:** average of per-class f1-scores.

## Results and Analysis:

The following results are averages over five runs for each experiment.

### Validation Results

| Model | Accuracy | Agree | Disagree | Discuss | Unrelated | Macro f1-score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 0.239 | 0.220 | 0.169 | 0.154 | 0.327 | 0.218 |
| Majority | 0.517 | 0.000 | 0.000 | 0.000 | 0.681 | 0.170 |
| MLP | **0.781** | **0.760** | **0.738** | **0.513** | **0.846** | **0.714** |
| CNN | 0.766 | 0.731 | 0.730 | 0.479 | 0.833 | 0.693 |
| BiLSTM | 0.737 | 0.699 | 0.657 | 0.465 | 0.822 | 0.661 |
| CNN-BiLSTM | 0.735 | 0.691 | 0.635 | 0.445 | 0.820 | 0.648 |
| BiLSTM-CNN | 0.740 | 0.712 | 0.667 | 0.443 | 0.832 | 0.663 |
| CNN+BiLSTM | 0.744 | 0.686 | 0.687 | 0.476 | 0.818 | 0.667 |

### Testing Results

| Model | Accuracy | Agree | Disagree | Discuss | Unrelated | Macro f1-score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 0.248 | 0.230 | 0.156 | 0.160 | 0.336 | 0.221 |
| Majority | 0.554 | 0.000 | 0.000 | 0.000 | 0.713 | 0.178 |
| MLP | 0.824 | 0.818 | **0.767** | 0.425 | 0.886 | **0.724** |
| CNN | **0.825** | **0.822** | 0.740 | 0.406 | **0.895** | 0.716 |
| BiLSTM | 0.792 | 0.774 | 0.710 | 0.392 | 0.869 | 0.686 |
| CNN-BiLSTM | 0.787 | 0.773 | 0.661 | 0.319 | 0.877 | 0.658 |
| BiLSTM-CNN | 0.794 | 0.772 | 0.701 | **0.459** | 0.874 | 0.702 |
| CNN+BiLSTM | 0.813 | 0.802 | 0.725 | 0.407 | 0.883 | 0.704 |

### Quick Analysis

* MLP achieves the highest macro F1-score across validation and testing sets.
* CNN outperforms BiLSTM across both sets.
* Combination models generally perform well, but do not outperform individual models.
* All models struggle with the "Discuss" class, indicating a difficulty in distinguishing it from other classes.

## Future Work:

It is possible to further improve the performance of the models on this classification task by carefully considering the following:

1. **Class Imbalance**: techniques like class weighting or oversampling could be explored to address this imbalance.
2. **Hyperparameter Tuning**: Conduct a thorough hyperparameter search to optimize the performance of each model.
3. **Transfer Learning**: Leverage pre-trained models from large language models or other relevant domains to improve performance.
