import torch
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CAU HINH - DIEU CHINH TAT CA TAI DAY
# ============================================================
NUM_CLASS    = 2
CLASS_NAMES  = ["healthy", "unhealthy"]
DEVICE       = 'cuda' if torch.cuda.is_available() else print('Cant execute because this model need to be run on GPU ')

from pathlib import Path
# Paths
PROJECT_DIR  = str(Path(__file__).resolve().parent.parent)
SAVE_DIR     = str(Path(PROJECT_DIR) / "output")

# Plot
FIGURE_SIZE  = (14, 5)
DPI          = 150

# ============================================================
# 1. Precision & Recall
# ============================================================

def compute_precision_recall(model, dataloader, device=DEVICE, num_class=NUM_CLASS):
    """
    Tinh Precision va Recall cho tung class.
    
    Args:
        model: model da train
        dataloader: validation/test dataloader
        device: 'cpu' hoac 'cuda'
        num_class: so luong class (mac dinh: 2)
    
    Returns:
        precision: dict {class_idx: precision_value}
        recall: dict {class_idx: recall_value}
        confusion_matrix: numpy array (num_class x num_class)
    """
    model.eval()
    model.to(device)

    # Confusion matrix: hang = actual, cot = predicted
    confusion = np.zeros((num_class, num_class), dtype=int)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t][p] += 1

    # Tinh Precision va Recall tu confusion matrix
    precision = {}
    recall = {}

    for c in range(num_class):
        # Precision = TP / (TP + FP)
        tp = confusion[c][c]
        fp = confusion[:, c].sum() - tp      # tong cot c - TP
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN)
        fn = confusion[c, :].sum() - tp       # tong hang c - TP
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall, confusion


def print_metrics(precision, recall, confusion, class_names=CLASS_NAMES):
    """In Precision, Recall va Confusion Matrix ra terminal."""
    num_class = len(precision)

    print("\n" + "=" * 45)
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10}")
    print("-" * 45)
    for c in range(num_class):
        print(f"  {class_names[c]:<15} {precision[c]:>10.4f} {recall[c]:>10.4f}")
    print("=" * 45)

    print("\nConfusion Matrix:")
    # Header
    header = f"  {'':>12}" + "".join(f"{class_names[c]:>12}" for c in range(num_class))
    print(header)
    for r in range(num_class):
        row = f"  {class_names[r]:>12}" + "".join(f"{confusion[r][c]:>12d}" for c in range(num_class))
        print(row)
    print()


# ============================================================
# 2. Accuracy & Loss Curve
# ============================================================

def plot_curves(history, model_name="model", save_dir=SAVE_DIR):
    """
    Ve bieu do Accuracy va Loss curve.

    Args:
        history: dict voi cac key:
            - 'train_acc': list accuracy training moi epoch
            - 'val_acc': list accuracy validation moi epoch
            - 'train_loss': list loss training moi epoch
            - 'val_loss': list loss validation moi epoch
        model_name: ten model (de dat tieu de va ten file)
        save_dir: thu muc luu file anh. Mac dinh: results/
    """
    epochs = range(1, len(history['train_acc']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    # --- Accuracy Curve ---
    ax1.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Train Accuracy')
    ax1.plot(epochs, history['val_acc'], 'r-o', markersize=4, label='Val Accuracy')
    ax1.set_title(f'{model_name} - Accuracy Curve', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Loss Curve ---
    ax2.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    ax2.plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
    ax2.set_title(f'{model_name} - Loss Curve', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filepath = f"{save_dir}\\{model_name}_curves.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Da luu bieu do tai: {filepath}")

    plt.show()


# ============================================================
# Test nhanh (chay doc lap de kiem tra)
# ============================================================
if __name__ == "__main__":
    # # --- Test plot_curves voi du lieu gia ---
    dummy_history = {
        'train_acc': [55, 65, 72, 78, 83, 87, 90, 92, 93, 94],
        'val_acc':   [50, 60, 68, 73, 77, 80, 82, 84, 85, 85],
        'train_loss': [1.2, 0.9, 0.7, 0.55, 0.42, 0.33, 0.25, 0.19, 0.15, 0.12],
        'val_loss':   [1.3, 1.0, 0.8, 0.65, 0.55, 0.48, 0.44, 0.42, 0.41, 0.40],
    }
    plot_curves(dummy_history, model_name="GoogLeNet_Test")

    # --- Test print_metrics voi du lieu gia ---
    dummy_precision = {0: 0.92, 1: 0.88}
    dummy_recall = {0: 0.90, 1: 0.91}
    dummy_confusion = np.array([[45, 5], [4, 46]])
    print_metrics(dummy_precision, dummy_recall, dummy_confusion)