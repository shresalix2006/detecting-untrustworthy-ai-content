# detecting-untrustworthy-ai-content
<p align="center"><img src="assets/banner.png" alt="Detecting Untrustworthy AI Content" width="100%"></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Hugging%20Face-DistilBERT-FFD21E?logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/Gradio-demo-F97316">
  <img src="https://img.shields.io/badge/Google%20Colab-notebook-F9AB00?logo=googlecolab&logoColor=white">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Deepfake%20test%20accuracy-95.2%25-2ea44f">
  <img src="https://img.shields.io/badge/Deepfake%20ROC--AUC-0.990-2ea44f">
  <img src="https://img.shields.io/badge/Hallucination%20F1-0.34-e0457b">
</p>

# 🛡️ Detecting Untrustworthy AI Content

Two neural-network detectors built and compared in one project:

1. **Deepfake detection:** a CNN that separates real photographs of faces from StyleGAN-generated fakes.
2. **Hallucination detection:** a fine-tuned DistilBERT transformer that flags LLM answers that are false or unsupported.

Both outputs look credible but cannot be trusted, so the project builds one automated detector per modality, judges them with the same imbalance-aware metrics, and compares how hard each task is.

Course project for Artificial Neural Networks, SRM Institute of Science and Technology (Delhi-NCR). The presentation with the literature survey and references is in [`slides.pdf`](slides.pdf).

## 👥 Team

Shrestha Chatterjee, Saurabh Yadav and Rajveer Chauhan. All code, training and evaluation: Shrestha Chatterjee. Literature survey and motivation: Rajveer Chauhan.

## 📊 Key results

| Task | Model | Main result |
|---|---|---|
| Deepfake face detection | ResNet-style CNN, trained from scratch | **95.2% test accuracy, ROC-AUC 0.990, F1 0.95** on 20,000 test images |
| Hallucination detection | Fine-tuned DistilBERT | **F1 0.34** on the hallucinated class (precision 0.27, recall 0.43) |

The contrast is the main finding: the image task is largely solved by a small CNN, while hallucination detection stays hard, even with class weighting.

## 🖼️ Pipeline 1: deepfake image detector

**Data.** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (Kaggle): 70,000 real faces and 70,000 StyleGAN-generated faces. Split into 100,000 train, 20,000 validation and 20,000 test images, perfectly balanced.

**Preprocessing.** Images are resized to 128 x 128 and pixel values scaled to 0-1.

**Model.** A ResNet-style CNN with residual blocks and a sigmoid output (real or fake probability), about 2.78 million parameters, trained from random initialisation (no pre-trained weights) with the AdamW optimiser and a cosine learning-rate schedule, using early stopping on validation accuracy. See the notebook for the exact settings.

**Test results (20,000 images).**

| Metric | Value |
|---|---|
| Accuracy | 95.2% |
| ROC-AUC | 0.990 |
| F1 (fake and real classes) | 0.95 |

Confusion matrix:

| | Predicted fake | Predicted real |
|---|---|---|
| **Actual fake** | 9,319 (93.2%) | 681 (6.8%) |
| **Actual real** | 272 (2.7%) | 9,728 (97.3%) |

The model leans slightly toward predicting "real": 681 fakes slipped through, against 272 real faces flagged as fake.

<p align="center"><img src="assets/deepfake_confusion.png" width="46%"></p>

**Further experiments in the notebook.**
- JPEG-compression robustness testing, with JPEG-augmentation fine-tuning.
- A transfer-learned ResNet50 comparison against the from-scratch model.
- Grad-CAM heatmaps to show what the detector looks at.
- A CIFAKE AI-generated-image detector.
- A Gradio demo that covers both detectors.

## 💬 Pipeline 2: hallucination detector

**Data.** Kaggle ML Olympiad "Detect hallucinations in LLMs": 16,687 labelled rows of (prompt, answer, target). Only 894 rows are hallucinated, so the data is heavily imbalanced (5.4% positive).

**Split.** Stratified 85/15 split: 14,183 train and 2,504 validation (134 hallucinated). Metrics are reported on the validation split because the competition's test file has no labels.

**Model.** `distilbert-base-uncased` with a new classification head (PyTorch, Hugging Face Trainer). Input is `prompt [SEP] answer`, truncated to 256 tokens. The loss is class-weighted cross-entropy (weight 0.53 for normal answers, 9.33 for hallucinated). Training ran for 3 epochs with batch size 16, and the best checkpoint was chosen by F1.

**Validation results.**

| Metric | Value |
|---|---|
| Accuracy | 90.8% |
| Precision (hallucinated) | 0.27 |
| Recall (hallucinated) | 0.43 |
| F1 (hallucinated) | 0.34 |

Confusion matrix:

| | Predicted normal | Predicted hallucinated |
|---|---|---|
| **Actual normal** | 2,216 | 154 |
| **Actual hallucinated** | 76 | 58 |

The model caught 58 of the 134 hallucinations, missed 76, and raised 154 false alarms.

<p align="center"><img src="assets/hallucination_confusion.png" width="46%"></p>

## ⚠️ Why accuracy alone is misleading

A "detector" that always answers "not hallucinated" scores **94.6% accuracy** and catches nothing. The trained model scores a lower 90.8% accuracy but actually finds hallucinations. That is why both detectors are judged with precision, recall, F1, ROC-AUC and a confusion matrix, not accuracy alone.

<p align="center"><img src="assets/why_accuracy_misleads.png" width="60%"></p>

## 🚧 Limitations

- The fakes are StyleGAN-only. Newer diffusion-based fakes were not tested, so the 95.2% should not be read as general deepfake detection.
- Single training run per model, with no hyperparameter search.
- Hallucination labels come from the dataset authors' heuristics and review, and there are only 894 positive examples.
- Hallucination metrics are on the validation split that was also used to choose the best checkpoint.

## 🔭 Future work

- Oversample the rare hallucination class and tune the decision threshold on a precision-recall curve.
- Try stronger image backbones (EfficientNet, ViT) and test on diffusion-based fakes and across datasets.
- Add retrieval or entailment checks against evidence for hallucination detection.
- Merge both detectors into one trust-score service.

## ▶️ Run it

1. Open `deepfake_and_hallucination_detection.ipynb` in Google Colab with a GPU runtime.
2. Install anything missing (`kagglehub`, `transformers`, `gradio`) and run the cells in order.
3. The notebook saves models to Google Drive (`/content/drive/MyDrive/deepfake/...`). Change these paths if you use a different folder.
4. The hallucination data needs `train.csv` from the Kaggle ML Olympiad competition.

> Note: the figures above come from the project presentation (`slides.pdf`). The notebook is shared without saved evaluation outputs, so re-run the evaluation cells to reproduce them.

## 📁 Files

- [`deepfake_and_hallucination_detection.ipynb`](deepfake_and_hallucination_detection.ipynb): both pipelines, extra experiments and the Gradio demo
- [`slides.pdf`](slides.pdf): project presentation, with literature survey and references

## 📚 Data sources

- xhlulu, "140k Real and Fake Faces", Kaggle dataset
- "ML Olympiad - Detect hallucinations in LLMs", Kaggle competition (Google ML community, 2024)
