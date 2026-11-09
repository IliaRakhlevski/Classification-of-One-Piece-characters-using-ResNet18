# Classification-of-One-Piece-characters-using-ResNet18

Image classification of One-Piece characters using transfer learning and ResNet18.

---

## Overview

This project implements an image classification pipeline for recognizing 18 different One-Piece characters using a pretrained ResNet18 model and transfer learning techniques.

The project includes:
- data preprocessing,
- image normalization,
- transfer learning with fine-tuning,
- model evaluation,
- confusion matrix analysis,
- visualization of training metrics,
- analysis of highly confident misclassifications.

The model was trained and evaluated using PyTorch.

---

## Dataset

The dataset contains 18 character classes from the One-Piece universe.

Each class contains approximately 650 images:
- original images,
- rotated images,
- color-inverted images,
- other transformed variations.

Dataset source:
- Public Kaggle dataset

The dataset is not included in this repository. The code expects it already split into three directories next to `main.py`, with one sub-directory per class (the names must match `CLASS_NAMES` in `main.py`):

```text
Train/Ace/..., Train/Akainu/..., ...
Val/Ace/...,   Val/Akainu/...,   ...
Test/Ace/...,  Test/Akainu/...,  ...
```

The split was prepared before training and is not performed by the code. Images derived from the same original image were kept in the same subset in order to reduce data leakage between training, validation, and test sets.

---

## Model Architecture

The project uses a pretrained ResNet18 model from `torchvision.models`.

Transfer learning strategy:
- freeze pretrained layers,
- replace final classification layer,
- fine-tune deeper layers (`layer3` and `layer4`).

Final classification layer:

```python
model.fc = nn.Linear(num_features, NUM_CLASSES)
```

Number of classes:
- 18

---

## Technologies

- Python 3.10
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Training Configuration

- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Scheduler: StepLR
- Epochs: 80
- Batch Size: 32

Training is seeded for reproducibility (`set_seed()` in `utils.py`). Runs were made with the seeds 42, 123 and 777; `main.py` currently uses 777, and the other two are kept as commented-out lines.

---

## Results

Best validation accuracy:
- 92.24%

Test accuracy:
- 91.72%

The test accuracy is measured with the model weights from the epoch with the best validation accuracy.

---

## Training Curves

### Accuracy

![Training Accuracy](Selected_Results/Train_Val_Acc.png)

### Loss

![Training Loss](Selected_Results/Train_Val_Losses.png)

---

## Confusion Matrix

Normalized confusion matrix:

![Confusion Matrix](Selected_Results/CM_plot.png)

---

## Highly Confident Misclassifications

Examples of incorrect predictions where the model was highly confident:

![Top Mistakes](Selected_Results/Top_confident_mistakes.png)

These examples help analyze model weaknesses and understand difficult edge cases.

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the `Train`, `Val` and `Test` directories next to `main.py` (see [Dataset](#dataset)), then run training and evaluation:

```bash
python main.py
```

The weights of the best epoch are saved to `best_model.pth`.

---

## Notes

This project focuses on:
- transfer learning,
- reproducibility,
- model evaluation,
- error analysis.

The goal was not only to achieve high accuracy, but also to understand model behavior and limitations.

---

## Repository Structure

```text
├── main.py
├── utils.py
├── visualization.py
├── requirements.txt
├── Project Description.docx
├── README.md
├── LICENSE
└── Selected_Results/
    ├── CM_plot.png
    ├── Top_confident_mistakes.png
    ├── Train_Val_Acc.png
    └── Train_Val_Losses.png
```

---

## Future Improvements

Possible future improvements:
- stronger data augmentation,
- larger backbone architectures,
- better handling of transformed images,
- ONNX export,
- real-time inference pipeline.

---

## License

MIT - see [LICENSE](LICENSE).
