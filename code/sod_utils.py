import torch
import torchvision as tv
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
# from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision as _MeanAveragePrecision
import kornia as K
from typing import List
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np

def binary_mask_to_instance_mask(mask, num_iterations=500) -> List[torch.Tensor]:
    """Separate a binary mask into multiple instance masks.
    We consider an instance to be a connected component.
    TODO: Verify this works with batched data.
    Args:
        mask: Binary masks of shape (1, 1, H, W)
        num_iterations: Number of iterations to run connected components.
            I would keep this number as high as possible, but it gets slow when it's too high.
            TODO: Tune this parameter for your case. I'm not sure what the best value is.
    Returns:
        List[torch.Tensor]: Instance masks for each example.
            Because the number of instances per example may be different,
            we have 
    """

    mask = torch.from_numpy(mask)
    # plt.imshow(mask, cmap="bone")
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 0)
    # print(mask.shape)
    mask = mask.type(torch.float32)
    labels = K.contrib.connected_components(mask, num_iterations=num_iterations)
    # print(labels)

    # Separate the labels into individual masks.
    batch_instance_masks = []
    for ex_label in labels:
        unique_labels = torch.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]   # Ignore background (0 class)
        instance_masks = torch.concat([torch.where(ex_label == label, 1, 0).type(torch.bool) for label in unique_labels], dim=0)  # Shape: (# instances, H, W)
        batch_instance_masks.append(instance_masks)
    return batch_instance_masks


class MeanAveragePrecision(_MeanAveragePrecision):
    """
    A custom implementation of the mean average precision metric for
    binary Long Range Salient Object Detection.

    Reference:
        https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
    """
    def __init__(
        self,
        box_format: str = "xyxy",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            box_format=box_format,
            # We will always use the segmentation IoU.
            iou_type="segm",
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            max_detection_thresholds=max_detection_thresholds,
            class_metrics=class_metrics,
            **kwargs
        )

        self.bbox_area_ranges = {
            "all": (0**2, int(1e5**2)),
            "small": (0**2, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, int(1e5**2)),
        }
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """The method to call to update the metric for each batch.

        Args:
            preds: Predicted binary masks of shape (batch, 1, H, W).
            target: Ground truth binary mask of shape (batch, 1, H, W)
        """
        # Convert the binary masks to instance masks
        pred_instance_masks = binary_mask_to_instance_mask(preds)
        target_instance_masks = binary_mask_to_instance_mask(target)

        # Convert the instance masks to bounding boxes.
        pred_bboxes = [masks_to_boxes(instance_masks) for instance_masks in pred_instance_masks]
        target_bboxes = [masks_to_boxes(instance_masks) for instance_masks in target_instance_masks]
        
        # Get the scores for each of the predictions.
        pred_scores = greedy_scores(pred_instance_masks, target_instance_masks)

        # Format the data in the way superclass expects it.
        preds = [
            {
                "boxes": pred_bboxes[i],
                "scores": pred_scores[i],
                "labels": torch.ones(len(pred_bboxes[i]), dtype=torch.int64),
                "masks": pred_instance_masks[i],
            }
            for i in range(len(pred_bboxes))
        ]

        target = [
            {
                "boxes": target_bboxes[i],
                "labels": torch.ones(len(target_bboxes[i]), dtype=torch.int64),
                "masks": target_instance_masks[i],
            }
            for i in range(len(target_bboxes))
        ]

        return super().update(preds, target)


def greedy_scores(pred_instance_masks, target_instance_masks):
    """Greedy score matching algorithm.

    All mean AP implementations require scores for each of the instance predictions.
    However, because we aren't doing detection in a conventional way, we don't have 
    a clear way of defining scores.

    As a stopgap, we greedily score the predictions based on the IoU between the
    predicted and ground truth instance masks.

    Args:
        pred_instance_masks: torch.Tensor]z of shape (batch, # pred instances, H, W)
        target_instance_masks: List[torch.Tensor] of shape (batch, # ground truth instances, H, W)
    
    Returns:
        List[torch.Tensor]: A list of example scores. Each example score is a tensor
            of the same length as the number of predictions for that example.
    """
    return [_greedy_score_example(pred, target) for pred, target in zip(pred_instance_masks, target_instance_masks)]


def _greedy_score_example(preds: torch.Tensor, targets: torch.Tensor):
    def _iou(_pred, _target):
        return torch.sum(_pred * _target) / (torch.sum(_pred + _target) + 1e-8)
    # This operation may be pretty expensive when the number of instances is large.
    # TODO: We may need to filter the number of predictions we want to score.
    ious = torch.as_tensor([
        [_iou(preds[i], targets[j]) for j in range(len(targets))]
        for i in range(len(preds))
    ])
    scores = torch.zeros(len(preds))
    while True:
        if torch.all(ious < 0):
            break
        # Find the highest IoU between predictions and ground truth.
        # This is the best match.
        best_idx = ious.argmax()
        pred_idx, target_idx = (torch.div(best_idx, len(ious[0]), rounding_mode="floor"), best_idx % len(ious[0]))
        # Assign the score to the best prediction.
        scores[pred_idx] = ious[pred_idx, target_idx]
        # Set the IOU for the prediction and the target to -1
        # so that they don't appear again when matching.
        # IOUs are always positive, so -1 suffices for not finding it in max.
        ious[pred_idx] = -1
        ious[:, target_idx] = -1

    return scores