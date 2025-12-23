import torch
import torch.nn as nn
import torch.nn.functional as F

from config.vars import IMG_SIZE, NUM_CLASSES


class SimpleCNNGAP(nn.Module):
    """
    Convolutional network with global average pooling.
    Almost translation invariant.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B,16,H,W)
        x = F.relu(self.conv2(x))  # (B,32,H,W)
        x = F.adaptive_avg_pool2d(x, 1)  # (B,32,1,1)
        x = x.view(x.size(0), -1)  # (B,32)
        x = self.classifier(x)  # (B,C)
        return x


class SimpleCNNFC(nn.Module):
    """
    Convolutional network with a fully connected head that sees all spatial positions.
    Has explicit access to absolute location.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * IMG_SIZE * IMG_SIZE, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleTransformerClassifier(nn.Module):
    """
    Simple transformer-based classifier on flattened image pixels
    with learned positional embeddings.
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.img_size = img_size
        self.seq_len = img_size * img_size

        # token embedding: scalar pixel -> d_model
        self.token_embed = nn.Linear(1, d_model)

        # positional embeddings
        self.pos_embed = nn.Embedding(self.seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classification head
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (B, 1, H, W)
        Flatten to sequence length L = H*W.
        """
        B = x.size(0)
        # (B,1,H,W) -> (B,H*W,1)
        x = x.view(B, 1, self.img_size * self.img_size).transpose(1, 2)

        # token embeddings: (B,L,1) -> (B,L,d_model)
        tok = self.token_embed(x)

        # positions 0..L-1
        device = x.device
        pos_ids = torch.arange(self.seq_len, device=device).unsqueeze(0).expand(B, -1)
        pos = self.pos_embed(pos_ids)  # (B,L,d_model)

        x = tok + pos  # (B,L,d_model)

        # transformer encoder
        x = self.encoder(x)  # (B,L,d_model)

        # mean pool across sequence
        x = x.mean(dim=1)  # (B,d_model)

        # class logits
        x = self.cls(x)
        return x


class HybridConvTransformer(nn.Module):
    """
    Hybrid model: convolutional stem to reduce spatial resolution,
    then transformer over patch tokens.

    - Conv stem: 1 -> 16 -> 32 channels with stride-2 downsamples (H/4 x W/4).
    - Transformer encoder on flattened (H/4 * W/4) tokens, each of dim=32->d_model.
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.img_size = img_size

        # conv stem: downsample by 4
        self.stem_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # H/2
        self.stem_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # H/4

        self.patch_h = img_size // 4
        self.patch_w = img_size // 4
        self.seq_len = self.patch_h * self.patch_w

        # project conv features (32-dim) to d_model
        self.proj = nn.Linear(32, d_model)

        # positional embeddings for tokens
        self.pos_embed = nn.Embedding(self.seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (B, 1, H, W)
        """
        B = x.size(0)
        x = F.relu(self.stem_conv1(x))  # (B,16,H/2,W/2)
        x = F.relu(self.stem_conv2(x))  # (B,32,H/4,W/4)

        # (B,32,H',W') -> (B,H'*W',32)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, self.seq_len, 32)

        # project to d_model
        x = self.proj(x)  # (B,seq_len,d_model)

        device = x.device
        pos_ids = torch.arange(self.seq_len, device=device).unsqueeze(0).expand(B, -1)
        pos = self.pos_embed(pos_ids)

        x = x + pos

        x = self.encoder(x)

        x = x.mean(dim=1)
        x = self.cls(x)
        return x
