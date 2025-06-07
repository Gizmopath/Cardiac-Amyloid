import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets.custom_dataset import CustomImageDataset
from models.efficientnet import get_efficientnet
from utils.helpers import get_images_from_cases
from utils.file_utils import load_test_cases_from_file
from utils.metrics import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
batch_size = 64
num_folds = 10

for fold in range(1, num_folds + 1):
    print(f'Evaluating Fold {fold}')
    test_cases = load_test_cases_from_file(f'results/fold_{fold}_cases.txt')
    test_labels = [1 if 'positive' in c.lower() else 0 for c in test_cases]
    test_image_paths, test_image_labels = get_images_from_cases(test_cases, test_labels)

    test_dataset = CustomImageDataset(test_image_paths, test_image_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_efficientnet().to(device)
    model.load_state_dict(torch.load(f'results/best_model_fold_{fold}.pth'))
    acc, f1, auc = evaluate_model(model, test_loader, device)

    print(f'Fold {fold} - Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
