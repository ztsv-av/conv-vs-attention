import torch
from torch.utils.data import Dataset, DataLoader

from vars import MARGIN, NUM_PIXELS, IMG_SIZE, NUM_CORNERS, NUM_CLASSES, TRAIN_SAMPLES, TEST_SAMPLES, BATCH_SIZE

if NUM_PIXELS >= IMG_SIZE:
    raise ValueError(f"NUM_PIXELS={NUM_PIXELS} must be < IMG_SIZE={IMG_SIZE}")

max_start = IMG_SIZE - NUM_PIXELS - MARGIN  # last valid top-left index

CORNERS = {
    0: (MARGIN, MARGIN),  # top-left
    1: (MARGIN, max_start),  # top-right
    2: (max_start, MARGIN),  # bottom-left
    3: (max_start, max_start),  # bottom-right
}


def generate_sample():
    mask = torch.randint(1, NUM_CLASSES + 1, (1,)).item()
    img = torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

    for corner_idx in range(NUM_CORNERS):
        if (mask >> corner_idx) & 1:
            r, c = CORNERS[corner_idx]
            img[0, r : r + NUM_PIXELS, c : c + NUM_PIXELS] = 1.0

    label = mask - 1
    return img, label


class CornerSubsetDataset(Dataset):
    def __init__(self, n_samples: int):
        self.images = []
        self.labels = []
        for _ in range(n_samples):
            img, lab = generate_sample()
            self.images.append(img)
            self.labels.append(lab)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_datasets_and_loaders():
    train_ds = CornerSubsetDataset(TRAIN_SAMPLES)
    test_ds = CornerSubsetDataset(TEST_SAMPLES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_ds, test_ds, train_loader, test_loader
