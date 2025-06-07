import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets.custom_dataset import CustomImageDataset
from models.efficientnet import get_efficientnet
from utils.helpers import get_images_from_cases
from utils.file_utils import load_test_cases_from_file
from utils.metrics import calculate_case_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
batch_size = 64
num_folds = 10

for fold in range(1, num_folds + 1):
    print(f'Evaluating Fold {fold}')
    test_cases = load_test_cases_from_file(f'results/fold_{fold}_cases.txt')

    model = get_efficientnet().to(device)
    model.load_state_dict(torch.load(f'results/best_model_fold_{fold}.pth'))
    model.eval()

    for case in test_cases:
        label = 1 if 'positive' in case.lower() else 0
        image_paths, labels = get_images_from_cases([case], [label])
        if not image_paths:
            continue

        dataset = CustomImageDataset(image_paths, labels, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds, gts = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()
                preds.extend(batch_preds)
                gts.extend(lbls.numpy())

        metrics = calculate_case_metrics(preds, gts, case)
        print(f"Case {metrics['case_label']} - TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
