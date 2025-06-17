import os
import json
import torch
import logging
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch import nn, optim

from classes.ShahedDataset import ShahedDataset, label2idx
from classes.ShahedNet import ShahedNet

from sklearn.metrics import f1_score, precision_score, recall_score

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def find_best_threshold(y_true, y_prob):
    thresholds = torch.linspace(0.0, 1.0, steps=101)
    best_f1 = 0.0
    best_threshold = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t.item()).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t.item()
    return best_threshold, best_f1

def save_thresholds_to_file(threshold_dict, txt_path="saved_models/class_thresholds.txt", json_path="saved_models/class_thresholds.json"):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "w") as f:
        min_thr, max_thr, mean_thr = [], [], []
        for label, info in threshold_dict.items():
            thr = info["threshold"]
            f1 = info["f1"]
            f.write(f"{label}: threshold={thr:.3f}, F1={f1:.4f}\n")
            min_thr.append(thr)
            max_thr.append(thr)
            mean_thr.append(thr)
        f.write("\nЗведена інформація:\n")
        f.write(f"Min threshold: {min(min_thr):.3f}\n")
        f.write(f"Max threshold: {max(max_thr):.3f}\n")
        f.write(f"Mean threshold: {sum(mean_thr)/len(mean_thr):.3f}\n")

    with open(json_path, "w") as jf:
        json.dump({k: v["threshold"] for k, v in threshold_dict.items()}, jf, indent=4)

def calculate_pos_weights(dataset):
    labels = [sample[1] for sample in dataset]
    labels_tensor = torch.stack(labels)
    total = labels_tensor.shape[0]
    pos_counts = labels_tensor.sum(dim=0)
    neg_counts = total - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-6)
    return pos_weights

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc=f"Train Epoch {epoch+1}", leave=False)
    for i, (waveforms, labels) in enumerate(loop):
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveforms.size(0)
        total += labels.size(0)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        loop.set_postfix(loss=running_loss / total)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return running_loss / total, f1, precision, recall

def valid_epoch(model, loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    val_total = 0

    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc=f"Valid Epoch {epoch+1}", leave=False)
    with torch.no_grad():
        for i, (waveforms, labels) in enumerate(loop):
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * waveforms.size(0)
            val_total += labels.size(0)

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())

            loop.set_postfix(val_loss=val_loss / val_total)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    f1 = f1_score(all_labels, all_preds > 0.5, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds > 0.5, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds > 0.5, average='macro', zero_division=0)

    threshold_info = {}
    for class_name, idx in label2idx.items():
        class_probs = all_preds[:, idx]
        class_true = all_labels[:, idx]
        best_thr, best_f1 = find_best_threshold(class_true, class_probs)
        threshold_info[class_name] = {"threshold": best_thr, "f1": best_f1}
        logger.info(f"[Поріг] {class_name}: threshold={best_thr:.2f}, F1={best_f1:.4f}")

    save_thresholds_to_file(threshold_info)

    return val_loss / val_total, f1, precision, recall

def main():
    epochs = 100
    batch_size = 16
    lr = 0.00001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Використовується пристрій: {device}")

    train_dataset = ShahedDataset(
        csv_path="dataset/mixed_train/labels.csv",
        audio_dir="dataset/mixed_train",
        label2idx=label2idx
    )

    valid_dataset = ShahedDataset(
        csv_path="dataset/mixed_valid/labels.csv",
        audio_dir="dataset/mixed_valid",
        label2idx=label2idx
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = ShahedNet(n_classes=len(label2idx)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0

    for epoch in range(epochs):
        logger.info(f"Епоха {epoch + 1}/{epochs}")
        train_loss, train_f1, train_prec, train_rec = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_f1, val_prec, val_rec = valid_epoch(model, valid_loader, criterion, device, epoch)

        logger.info(
            f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f} | "
            f"Valid Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "saved_models/shahednet_best.pth")
            logger.info(f"Збережено кращу модель на епосі {epoch + 1} з F1={val_f1:.4f}")

if __name__ == "__main__":
    main()
