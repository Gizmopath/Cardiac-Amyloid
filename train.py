import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from datasets.custom_dataset import CustomImageDataset
from models.efficientnet import get_efficientnet
from utils.helpers import get_cases_and_labels, get_images_from_cases

# General settings
positive_folder = "data/positive"
negative_folder = "data/negative"
num_epochs = 1
batch_size = 128
learning_rate = 0.001
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_folds = 10

transform_train = T.Compose([
    T.RandomApply([T.Lambda(lambda img: img.rotate(torch.randint(1, 4, ()).item() * 90))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.6, saturation=0.6, hue=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 3.0))], p=0.3),
    T.Resize((256, 256)),
    T.ToTensor(),
])

transform_val = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

cases, case_labels = get_cases_and_labels(positive_folder, negative_folder)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_val_idx, test_idx) in enumerate(kf.split(cases)):
    print(f'Fold {fold + 1}/{num_folds}')
    train_val_cases = [cases[i] for i in train_val_idx]
    train_val_labels = [case_labels[i] for i in train_val_idx]
    test_cases = [cases[i] for i in test_idx]
    test_labels = [case_labels[i] for i in test_idx]

    train_cases, val_cases, train_labels, val_labels = train_test_split(train_val_cases, train_val_labels, test_size=0.1111, stratify=train_val_labels, random_state=42)

    train_image_paths, train_labels = get_images_from_cases(train_cases, train_labels)
    val_image_paths, val_labels = get_images_from_cases(val_cases, val_labels)

    train_dataset = CustomImageDataset(train_image_paths, train_labels, transform=transform_train)
    val_dataset = CustomImageDataset(val_image_paths, val_labels, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_efficientnet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader
            running_loss, all_preds, all_labels = 0.0, [], []

            with tqdm(total=len(dataloader), desc=f"{phase.capitalize()} Epoch {epoch+1}") as pbar:
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = outputs.argmax(dim=1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    pbar.update(1)

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            auc = roc_auc_score(all_labels, all_preds)
            print(f'{phase.capitalize()} Loss: {running_loss:.4f} Acc: {acc:.4f} F1: {f1:.4f} AUC: {auc:.4f}')

    torch.save(model.state_dict(), f"results/best_model_fold_{fold+1}.pth")
    with open(f"results/fold_{fold+1}_cases.txt", "w") as f:
        f.write("Test Cases:\n" + "\n".join(test_cases))
