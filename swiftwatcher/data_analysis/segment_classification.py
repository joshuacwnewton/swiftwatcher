"""
    Contains functionality to classify segmented objects within frames.
    Used to prevent non-chimney swift objects (e.g. seagulls)
    from being treated as chimney swifts.
"""

import torch
from torch import nn
from torchvision import models, transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SegmentClassifier:
    def __init__(self, model_path):
        self.model = setup_model(2)
        self.model.load_state_dict(torch.load(model_path))
        self.transforms = [
            transforms.ToPILImage(),
            transforms.Resize((24, 24)),
            transforms.Pad((224 - 24)//2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    def __call__(self, segments):
        segments_to_keep = []

        for segment in segments:
            x = segment.segment_image
            for transform in self.transforms:
                x = transform(x)
            x = x.unsqueeze(0)
            x = x.to(device)
            score = self.model(x)
            _, y_pred = torch.max(score, 1)

            if y_pred == 1:
                segments_to_keep.append(segment)

        for i, segment in enumerate(segments_to_keep):
            segment.label = i+1

        return segments_to_keep


def setup_model(num_classes):
    """Select CNN architecture, modified for transfer learning on
    specific dataset."""
    # Call model constructor
    model = models.squeezenet1_0(pretrained=True)

    # Freeze layer parameters if feature extracting
    for param in model.parameters():
        param.requires_grad = False

    # Modify model to support fewer classes: (512, 1000) -> (512, 2)
    # This will also unfreeze this layer's parameters, as the default value for
    # (weight/bias).required_grad = True
    model.classifier[1] = nn.Conv2d(512, num_classes,
                                    kernel_size=1)
    model.num_classes = num_classes

    # Send the model to GPU
    model = model.to(device)

    return model
