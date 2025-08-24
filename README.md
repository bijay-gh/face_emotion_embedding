# Face Emotion Embedding

This repository provides PyTorch implementations for training Swin Transformer-based models (with and without Squeeze-and-Excitation blocks) on facial emotion datasets such as FER2013 and RAF-DB. It supports Sharpness-Aware Minimization (SAM) for robust optimization.

## Features

- **Swin Transformer Backbone**: State-of-the-art vision transformer for image classification.
- **Squeeze-and-Excitation (SE) Block**: Optional channel attention for improved performance.
- **Sharpness-Aware Minimization (SAM)**: Advanced optimizer for better generalization.
- **Configurable Datasets**: Easily switch between FER2013, RAF_DB Single Emotion, and RAF_DB Multi Emotion via `config.yaml`.
- **Training & Evaluation**: Includes functions for training, validation, and test evaluation (accuracy & F1 score).
- **Best Model Saving**: Automatically saves the best model based on validation accuracy.

## Folder Structure

```
face_emotion_embedding/
│
├── train_swin.py         # Main training script
├── model.py              # Model definitions and training/evaluation functions
├── config.yaml           # Dataset paths and hyperparameters
├── output/               # Saved best model
└── README.md             # Project documentation
```

## Setup

1. **Clone the repository**
2. **Install dependencies**
    ```bash
    pip install torch torchvision torch-optimizer scikit-learn pyyaml
    ```

3. **Prepare your datasets**
   - Organize FER2013 and RAF_DB datasets as specified in `config.yaml`.

4. **Configure paths and hyperparameters**
   - Edit `config.yaml` to set correct dataset paths and training parameters.

## Usage

### Training

Run the training script:
```bash
python train_swin.py
```

### Configuration

Example `config.yaml`:
```yaml
dataset:
  FER2013:
    train: e:\Projects\Emotion_Classification\Dataset\FER2013\train
    test: e:\Projects\Emotion_Classification\Dataset\FER2013\test
  RAF_DB:
    Single_Emotion:
      train: e:\Projects\Emotion_Classification\Dataset\RAF_DB\Single_Emotion\train
      test: e:\Projects\Emotion_Classification\Dataset\RAF_DB\Single_Emotion\test
    Multi_Emotion:
      train: e:\Projects\Emotion_Classification\Dataset\RAF_DB\Multi_Emotion\train
      test: e:\Projects\Emotion_Classification\Dataset\RAF_DB\Multi_Emotion\test
lr: 0.001
epochs: 30
```

### Model Selection

- **Swin**: Standard Swin Transformer
- **SwinWithSE**: Swin Transformer with Squeeze-and-Excitation block

### Training with SAM

To use SAM optimizer, ensure `torch-optimizer` is installed and use the provided `train_one_epoch_sam` function.

## Evaluation

After training, the best model is saved in the `output` folder. The script prints test accuracy and F1 score.

## Citation

If you use this code, please cite as  
G. K. Mishra and B. Ghimire, "Typing with emotions: Emotion Embedding in text apps," TechRxiv, Aug. 2, 2025. DOI: 10.36227/techrxiv.175416003.30236370/v1

For the concept, please cite the original Swin Transformer and SAM papers.

## License

This project is licensed under the MIT License.