from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_preds)
    return acc, f1, auc

def calculate_case_metrics(preds, labels, case_label):
    tp = tn = fp = fn = 0
    for true, pred in zip(labels, preds):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    return {'case_label': case_label, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
