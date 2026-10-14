"""
object_classifier.py — standalone object-identity classifier.

Separate from the diffusion policy entirely: a small ResNet18-based image
classifier that answers "which of the N known objects is this?" from a
single global-camera frame. Exists to give the diffusion policy an explicit,
reliable object-identity signal (see model.py/dataset.py's 'object_id'
state_keys component) instead of forcing it to re-infer object identity from
a pooled vision feature vector at every independent replanning step, which
is the likely cause of trajectories drifting mid-rollout toward the wrong
object (see discussion in this project's notes: short per-frame conditioning
signals are susceptible to receding-horizon mode-switching, while a signal
computed once and held fixed for the whole episode is not).

Classifier checkpoint format (see train_object_classifier.py / ObjectClassifierInference):
    {
        'state_dict':  model state dict,
        'object_names': [str, ...],   # index i -> object name, defines num_objects
        'image_size':  (H, W),        # must match crop_and_resize's target
        'crop_scale':  float,
        'crop_x':      float,
        'crop_y':      float,
    }
"""
from __future__ import annotations

import sys
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware.image_utils import crop_and_resize


class ObjectClassifier(nn.Module):
    """ResNet18 (ImageNet-pretrained) + dropout + a single linear head.

    Deliberately simple: with a small number of visually-distinct object
    classes and pretrained features, a single linear layer on top is enough
    -- no need for a deeper head that would just risk overfitting the
    modest per-object episode count.
    """

    def __init__(self, num_objects: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.num_objects = num_objects
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(resnet.fc.in_features, num_objects),
        )

    def forward(self, x):
        """x : (B, 3, H, W) -> logits (B, num_objects)"""
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def preprocess_image(image_raw_uint8: np.ndarray, image_size, crop_scale: float,
                      crop_x: float, crop_y: float) -> np.ndarray:
    """uint8 (H,W,3) raw camera frame -> float32 (3,H,W) in [-1,1], same
    crop/resize/normalization convention as dataset.py / the deploy script's
    _preprocess_image, so the classifier sees exactly the same kind of crop
    the policy itself is trained/deployed on."""
    img = crop_and_resize(image_raw_uint8, image_size, crop_scale=crop_scale,
                           crop_x=crop_x, crop_y=crop_y)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img.transpose(2, 0, 1)   # (3,H,W)


class ObjectClassifierInference:
    """Loads a trained ObjectClassifier checkpoint and classifies frames.

    Used both standalone (e.g. for a quick accuracy check) and from the
    deployment script, which calls predict_onehot() ONCE at the start of
    each episode and holds the result fixed for the whole rollout.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.object_names = list(ckpt['object_names'])
        self.num_objects = len(self.object_names)
        self.image_size = tuple(ckpt['image_size'])
        self.crop_scale = float(ckpt['crop_scale'])
        self.crop_x = float(ckpt['crop_x'])
        self.crop_y = float(ckpt['crop_y'])

        self.model = ObjectClassifier(self.num_objects, pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()
        print(f"[object_classifier] Loaded: {checkpoint_path}")
        print(f"[object_classifier] object_names={self.object_names}")

    @torch.no_grad()
    def predict_logits(self, image_raw_uint8: np.ndarray) -> np.ndarray:
        """One frame -> (num_objects,) raw logits."""
        img = preprocess_image(image_raw_uint8, self.image_size, self.crop_scale,
                                self.crop_x, self.crop_y)
        x = torch.from_numpy(img).float().unsqueeze(0).to(self.device)  # (1,3,H,W)
        return self.model(x).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_onehot(self, image_raw_uint8_frames: Sequence[np.ndarray]) -> tuple[np.ndarray, int]:
        """
        Classify one or more frames (e.g. a few consecutive early frames of
        an episode) and return a MAJORITY-VOTE one-hot result -- a single
        frame can be passed as a length-1 sequence.

        Returns:
            onehot    : (num_objects,) float32, exactly one 1.0
            class_idx : the winning class index (same info, for logging)
        """
        votes = [int(np.argmax(self.predict_logits(f))) for f in image_raw_uint8_frames]
        class_idx = int(np.bincount(votes, minlength=self.num_objects).argmax())
        onehot = np.zeros(self.num_objects, dtype=np.float32)
        onehot[class_idx] = 1.0
        return onehot, class_idx
