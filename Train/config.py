import os

# General settings
positive_folder = "/path/to/positive/images"
negative_folder = "/path/to/negative/images"
num_epochs = 20
batch_size = 64
learning_rate = 0.001
validation_split = 0.1
num_classes = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
patience = 5  # Early stopping patience

