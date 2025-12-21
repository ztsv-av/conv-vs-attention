import torch
from torch.utils.data import Dataset, DataLoader

from vars import IMG_SIZE, NUM_CORNERS, NUM_CLASSES, TRAIN_SAMPLES, TEST_SAMPLES, BATCH_SIZE

# corner positions (row, col)
CORNERS = {
    0: (2, 2),  # top-left
    1: (2, IMG_SIZE - 3),  # top-right
    2: (IMG_SIZE - 3, 2),  # bottom-left
    3: (IMG_SIZE - 3, IMG_SIZE - 3),  # bottom-right
}


def generate_sample():
    """
    Returns:
        img: FloatTensor (1, H, W), white (0) with 1..4 pixels set to 1 at corners.
        label: int in [0..NUM_CLASSES-1], representing a non-empty subset of corners as a bitmask.
               Internal mask is in [1..2^4-1], label = mask - 1.
    """
    # pick a non-empty subset of corners (bitmask in [1, 15])
    mask = torch.randint(1, NUM_CLASSES + 1, (1,)).item()  # 1..15
    img = torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

    for corner_idx in range(NUM_CORNERS):
        if (mask >> corner_idx) & 1:
            r, c = CORNERS[corner_idx]
            img[0, r, c] = 1.0

    label = mask - 1  # shift to 0..14
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
