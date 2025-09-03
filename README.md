# 🎙️ Voice-Recognizer

This project uses machine learning to classify gender based on voice characteristics. By analyzing acoustic features like frequency, entropy, and modulation, the model predicts whether a voice belongs to a male or female speaker.

📌 Objective

Train a supervised classification model to recognize gender from voice recordings using acoustic signal features.

📂 Dataset

Source: voice.csv

Target Variable: label (binary: male or female)

Features: Includes meanfreq, centroid, sp.ent, modindx, and other spectral properties

Preprocessing:

Removed missing and duplicate entries

Encoded target labels (male → 0, female → 1)

Scaled features using StandardScaler

🧠 Model: Support Vector Machine (SVM)

SVM was selected for its robustness in handling high-dimensional, non-linear data—ideal for voice-based classification tasks.

🔍 Hyperparameter Tuning:

RandomizedSearchCV

To optimize performance, we used RandomizedSearchCV to tune:

C: Regularization parameter

kernel: 'linear', 'rbf', 'poly'

gamma: 'scale', 'auto'

python
param_dist = {
    'C': np.logspace(-2, 2, 10),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
Iterations: 20 random combinations

Cross-validation: 3-fold

Scoring metric: Accuracy

📊 Evaluation

Confusion Matrix: Visualized with ConfusionMatrixDisplay

Classification Report: Precision, recall, F1-score

Accuracy: Evaluated on test set

✅ Results

Achieved high accuracy with tuned SVM

Best parameters selected via RandomizedSearchCV

Model generalizes well to unseen voice samples

🚀 How to Run

python VoiceRecognizer.py
Ensure the dataset is in the same directory or update the path accordingly.

📌 Future Enhancements

Test other models like Random Forest or XGBoost

Add voice signal visualizations (e.g., spectrograms)

Deploy as a web app for real-time gender prediction
