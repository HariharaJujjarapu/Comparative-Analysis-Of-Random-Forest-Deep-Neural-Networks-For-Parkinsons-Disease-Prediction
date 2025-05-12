# 🧠 Parkinson's Disease Detection using Deep Learning

This project implements a deep learning model to detect Parkinson's disease using biomedical voice measurements. The model is trained on the Parkinson's Disease dataset and leverages a fully connected neural network to classify patients based on extracted features.

## 📊 Dataset
- **Source:** [Parkinson's Disease Dataset](https://drive.google.com/file/d/1yz1PLhXWBccsKdv5JW1sqkpgumfUwvXg/view?usp=sharing)
- **Features:**
  - 22 biomedical voice measures per instance
  - Binary target: status (1 = Parkinson's, 0 = Healthy)
- **Preprocessing:**
  - Removed irrelevant columns
  - Normalized features using MinMaxScaler


## ⚙️ Training Details
- **Architecture:** Fully Connected Neural Network (Dense layers)
- **Activation Functions:** ReLU (hidden layers), Sigmoid (output)
- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- **Batch Size:** 16
- **Epochs:** 100 (configurable)

> The model is saved as parkinsons_model.h5 and can be reused for inference on new samples..

## 📈 Output
- Achieved high accuracy on test data
- Model can distinguish Parkinson’s vs. healthy patients based on voice features

## How to Run the Project
1. **Install dependencies:**
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib
2. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/parkinsons-detection.git
   cd parkinsons-detection
3. **Run the notebook or script**
   - Use code.ipynb to explore training, evaluation, and visualization.
   - Or run a Python script version (if provided) for training/inference.
4. **Load and test saved model:**
   ```bash
   from tensorflow.keras.models import load_model
   model = load_model("parkinsons_model.h5")
