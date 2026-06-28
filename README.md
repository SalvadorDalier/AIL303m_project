# AIL303m Project — Fruit Quality Classification

Binary image classification of fruit as **healthy** or **unhealthy** using deep learning models with transfer learning.

## Project Structure

```
AIL303m_project/
├── data/
│   ├── raw_image/              # Original unprocessed images
│   ├── preprocessed/           # Preprocessed images
│   ├── train/                  # Training set
│   │   ├── healthy/
│   │   └── unhealthy/
│   └── valid/                  # Validation set
│       ├── healthy/
│       └── unhealthy/
├── src/
│   ├── gradcam.py              # Model benchmarking & Grad-CAM utilities
│   └── model/
│       ├── googlenet22.py      # GoogLeNet (Inception v1) model definition
│       └── densenet121.py      # DenseNet-121 model definition
├── weights/
│   └── googlenet.npy           # Saved model weights
├── refer/
│   └── GoogLeNet-Inception/    # Reference materials for GoogLeNet architecture
├── synthetic-fruit-image-generator/  # CGAN-based synthetic data generation
│   ├── generate_lemons.py      # Script to generate synthetic lemon images
│   ├── cgan_generator_*.h5     # Pre-trained CGAN generator checkpoints
│   ├── comparison.png          # Real vs synthetic comparison
│   └── gradcam.png             # Grad-CAM visualization output
├── requirement.txt             # Python dependencies
└── README.md
```

## Models

### GoogLeNet (Inception v1)

- **File:** `src/model/googlenet22.py`
- Pretrained on ImageNet (`GoogLeNet_Weights.DEFAULT`)
- Final fully-connected layer replaced for binary classification (`num_class=2`)
- Auxiliary logits disabled
- Input size: `(batch, 3, 224, 224)`

### DenseNet-121

- **File:** `src/model/densenet121.py`
- Pretrained on ImageNet (`DenseNet121_Weights.DEFAULT`)
- Classifier layer replaced for binary classification (`num_class=2`)
- Input size: `(batch, 3, 224, 224)`

## Dataset

Binary classification with two classes:

| Class       | Description                |
|-------------|----------------------------|
| `healthy`   | Good-quality fruit images  |
| `unhealthy` | Damaged/defective fruit images |

Data is split into `train/` and `valid/` directories, each containing `healthy/` and `unhealthy/` subfolders.

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate virtual environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### Key Dependencies

| Package        | Purpose                         |
|----------------|---------------------------------|
| `torch`        | PyTorch deep learning framework |
| `torchvision`  | Pretrained models & transforms  |
| `torchinfo`    | Model architecture summaries    |
| `tensorflow`   | CGAN synthetic data generator   |
| `matplotlib`   | Plotting & visualization        |
| `pillow`       | Image loading & processing      |

## Usage

### View model architecture

```bash
# GoogLeNet summary
python src/model/googlenet22.py

# DenseNet-121 summary
python src/model/densenet121.py
```

### Train a model

You can train a model using `train.py` inside the `src/` directory.

```powershell
cd src
# Train GoogLeNet (Inception v1) for 10 epochs
python train.py --model googlenet22 --epoch 10 --batch 32

# Train DenseNet-121 for 15 epochs with a progress bar
python train.py --model densenet121 --epoch 15 --batch 16 --verbose 2
```

Arguments:
- `--model`: Choose model (`googlenet22` or `densenet121`). Default is `googlenet22`.
- `--epoch`: Number of training epochs. Default is `10`.
- `--batch`: Batch size. Default is `32`.
- `--worker`: Number of worker threads for dataloader. Default is `2`.
- `--verbose`: Display mode: `0` (silent), `1` (epoch summaries), `2` (detailed progress bar). Default is `2`.

### Evaluate models

To evaluate the trained models on the validation dataset:

```powershell
cd src
python evaluate.py
```

### Run benchmark / Grad-CAM

```powershell
cd src
python gradcam.py
```

> **Note:** Scripts inside `src/` (`train.py`, `evaluate.py`, `gradcam.py`) use relative imports, so they must be run from the `src/` directory.

### Generate synthetic fruit images

```bash
python synthetic-fruit-image-generator/generate_lemons.py
```

## Synthetic Data Generation

The project includes a **Conditional GAN (CGAN)** pipeline for generating synthetic fruit images to augment the training dataset. Pre-trained generator checkpoints are provided at different training epochs:

- `cgan_generator_500.h5`
- `cgan_generator_1000.h5`
- `cgan_generator_1500.h5`

## Tracking Experiments

We keep a diary/log of all training runs, model configurations, hyperparameters, and results to monitor progress and avoid redundant experiments.

Please document every training attempt in:
- **File:** [`experiment_log.md`](experiment_log.md)

### What to record:
1. **Date & Time:** When the run was executed.
2. **Model Details:** Model name, configuration modifications.
3. **Hyperparameters:** Batch size, learning rate (LR), total epochs, optimizer type.
4. **Metrics:** Best Train/Val Loss, best Train/Val Accuracy.
5. **Notes (Crucial):** Observations (e.g., overfitting behavior, training speed, anomalies, potential next steps).

---

## License

This project is for academic use as part of the **AIL303m** course.

