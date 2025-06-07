import os

def get_cases_and_labels(positive_folder, negative_folder):
    positive_cases = [os.path.join(positive_folder, c) for c in os.listdir(positive_folder) if os.path.isdir(os.path.join(positive_folder, c))]
    negative_cases = [os.path.join(negative_folder, c) for c in os.listdir(negative_folder) if os.path.isdir(os.path.join(negative_folder, c))]
    cases = positive_cases + negative_cases
    labels = [1] * len(positive_cases) + [0] * len(negative_cases)
    return cases, labels

def get_images_from_cases(cases, labels):
    image_paths = []
    image_labels = []
    for case, label in zip(cases, labels):
        for fname in os.listdir(case):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_paths.append(os.path.join(case, fname))
                image_labels.append(label)
    return image_paths, image_labels
