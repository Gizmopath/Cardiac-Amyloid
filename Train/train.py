import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from config import *
from dataset import CustomImageDataset, get_augmentation_pipeline, get_val_transform
from model import create_model

# Prepare datasets
positive_images = [os.path.join(positive_folder, img) for img in os.listdir(positive_folder) if img.endswith(".jpg")]
negative_images = [os.path.join(negative_folder, img) for img in os.listdir(negative_folder) if img.endswith(".jpg")]

transform_train = get_augmentation_pipeline()
transform_val = get_val_transform()

dataset = CustomImageDataset(positive_images, negative_images, transform=transform_train)

# Split train/validation datasets
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create model
model = create_model().to(device)

# Loss function, optimizer, and scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# Training and Validation Loop
best_f1 = 0.0
best_model_wts = None
early_stop_counter = 0

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []

        with tqdm(total=len(dataloader), desc=f"{phase.capitalize()} Epoch {epoch}") as pbar:
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.update(1)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

        # Validation improvements
        if phase == 'val':
            scheduler.step(epoch_f1)
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Save the best model
model.load_state_dict(best_model_wts)

# Save TorchScript model
model.eval()
example_input = torch.randn(1, 3, 256, 256).to(device)
scripted_model = torch.jit.script(model)
torchscript_model_path = "torchscript_model18.pt"
scripted_model.save(torchscript_model_path)
print(f"TorchScript model saved at {torchscript_model_path}")

