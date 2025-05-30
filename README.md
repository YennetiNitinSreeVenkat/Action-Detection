# 🧠 Human Action Recognition using CNN + RNN

This project implements a Human Action Recognition (HAR) system that classifies actions in videos using a hybrid deep learning architecture: **CNN (Inception-V3)** for spatial feature extraction and **LSTM (RNN)** for temporal modeling. The model is trained and evaluated on the UCF101 dataset.

---

## 📁 Dataset

We use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), a widely used action recognition dataset with 13,320 videos categorized into 101 different human action classes. The dataset contains actions like sports, musical instrument playing, and human interactions.

---

## 🚀 Getting Started

### 🔧 Method 1: Run Locally

1. Clone this repository.
2. Download and extract the UCF101 dataset.
3. Run `create_train_test.py` to generate train-test split folders (based on the official split files).
4. Open `FinalCVProject.ipynb` in Jupyter Notebook.
5. Modify the dataset path in the notebook.
6. Execute cells step-by-step to train and evaluate the model.

### ☁️ Method 2: Run on Google Colab

1. Download the preprocessed dataset from [Google Drive](https://drive.google.com/open?id=1CB43hwAikzUHqUREeZ6cAwdo26PtoK_l).
2. Mount your Google Drive in Colab.
3. Open `FinalCVProject.ipynb` on Colab.
4. Run all cells — since features are pre-extracted, it runs faster.

---

## 🧪 Methodology

- **Data Preparation**: Videos are split into frames and organized into `train` and `test` folders.
- **Feature Extraction**: Each frame is passed through a pre-trained **Inception-V3** network (without the top layer) to extract spatial features.
- **Sequence Modeling**: The extracted features are input to a custom **LSTM**-based RNN to learn temporal dynamics.
- **Training**: The model is trained with categorical cross-entropy loss and Adam optimizer. We save the best model based on validation loss.
- **Inference**: For classification, frames of a video are passed through Inception-V3 + LSTM to predict the action class.

---

## 📊 Results

| Model Variant     | Accuracy (Validation) | Loss (Validation) |
|------------------|-----------------------|-------------------|
| CNN Only         | ~25%                  | ~7.2              |
| CNN + LSTM (RNN) | ~74%                  | ~1.0              |

### 📈 Training Graphs

![Training Accuracy](https://raw.githubusercontent.com/cynicphoenix/Human-Action-Recognition/master/Screenshots/training_aacuracy.png)
![Training Loss](https://raw.githubusercontent.com/cynicphoenix/Human-Action-Recognition/master/Screenshots/training_loss.png)
![Validation Accuracy](https://raw.githubusercontent.com/cynicphoenix/Human-Action-Recognition/master/Screenshots/validation_accuracy.png)
![Validation Loss](https://raw.githubusercontent.com/cynicphoenix/Human-Action-Recognition/master/Screenshots/validation_loss.png)

---

## 🧠 Conclusion

This project demonstrates that incorporating temporal dynamics with RNNs significantly improves action recognition performance over static CNN-only models. Although the current accuracy (~74%) is below the state-of-the-art (~94%), it shows promise and can be enhanced further with improved techniques and additional resources.

---

## 👨‍💻 Authors

- [Yenneti Nitin Sree Venkat](https://github.com/YennetiNitinSreeVenkat)

---

## 📎 Related Links

- Original Repository: [cynicphoenix/Human-Action-Recognition](https://github.com/cynicphoenix/Human-Action-Recognition)
- Dataset: [UCF101 Official Site](https://www.crcv.ucf.edu/data/UCF101.php)
