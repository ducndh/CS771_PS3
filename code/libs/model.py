import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn
from torch.nn import functional as F

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        cls_logits = []
        for feature in x:
            cls_logits.append(self.cls_logits(self.conv(feature)))
        return cls_logits


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        bbox_reg = []
        bbox_ctrness = []
        for feature in x:
            conv_out = self.conv(feature)
            bbox_reg.append(self.bbox_reg(conv_out))
            bbox_ctrness.append(self.bbox_ctrness(conv_out))
        return bbox_reg, bbox_ctrness


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function depends on if the model is in training
    or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detection results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)
    * You might want to double check the format of 2D coordinates saved in points

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        num_levels = len(points)

        all_labels = []
        all_bbox_targets = []
        all_ctr_targets = []

        for level_idx in range(num_levels):
            points_per_level = points[level_idx]
            stride = strides[level_idx]
            reg_range_per_level = reg_range[level_idx]

            H, W = points_per_level.shape[:2]
            points_flat = points_per_level.reshape(-1, 2)

            labels_per_img = []
            bbox_targets_per_img = []
            ctr_targets_per_img = []

            for img_idx, targets_per_img in enumerate(targets):
                gt_boxes = targets_per_img["boxes"]
                gt_labels = targets_per_img["labels"]

                num_points = points_flat.size(0)
                num_gt = gt_boxes.size(0)

                if num_gt == 0:
                    labels_per_img.append(torch.zeros(num_points, dtype=torch.int64, device=gt_boxes.device))
                    bbox_targets_per_img.append(torch.zeros(num_points, 4, device=gt_boxes.device))
                    ctr_targets_per_img.append(torch.zeros(num_points, device=gt_boxes.device))
                    continue

                points_expanded = points_flat.unsqueeze(1).expand(num_points, num_gt, 2)
                gt_boxes_expanded = gt_boxes.unsqueeze(0).expand(num_points, num_gt, 4)

                left = points_expanded[:, :, 0] - gt_boxes_expanded[:, :, 0]
                top = points_expanded[:, :, 1] - gt_boxes_expanded[:, :, 1]
                right = gt_boxes_expanded[:, :, 2] - points_expanded[:, :, 0]
                bottom = gt_boxes_expanded[:, :, 3] - points_expanded[:, :, 1]

                bbox_targets = torch.stack([left, top, right, bottom], dim=2)

                is_in_boxes = bbox_targets.min(dim=2)[0] > 0

                max_reg_targets = bbox_targets.max(dim=2)[0]
                is_in_range = (max_reg_targets >= reg_range_per_level[0]) & (max_reg_targets <= reg_range_per_level[1])

                is_valid = is_in_boxes & is_in_range

                gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
                gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
                gt_centers = torch.stack([gt_cx, gt_cy], dim=1)

                gt_centers_expanded = gt_centers.unsqueeze(0).expand(num_points, num_gt, 2)
                center_dists = torch.norm(points_expanded - gt_centers_expanded, dim=2)

                is_in_center = center_dists <= (self.center_sampling_radius * stride)

                is_positive = is_valid & is_in_center

                areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                areas_expanded = areas.unsqueeze(0).expand(num_points, num_gt)
                areas_expanded[~is_positive] = float('inf')

                min_area_inds = areas_expanded.min(dim=1)[1]

                labels = gt_labels[min_area_inds]
                labels[areas_expanded.min(dim=1)[0] == float('inf')] = 0

                bbox_targets_assigned = bbox_targets[torch.arange(num_points), min_area_inds]
                bbox_targets_assigned = bbox_targets_assigned / stride

                left_t = bbox_targets_assigned[:, 0]
                top_t = bbox_targets_assigned[:, 1]
                right_t = bbox_targets_assigned[:, 2]
                bottom_t = bbox_targets_assigned[:, 3]

                ctr_targets = torch.zeros_like(left_t)
                pos_inds = labels > 0
                if pos_inds.sum() > 0:
                    ctr_targets[pos_inds] = torch.sqrt(
                        (torch.min(left_t[pos_inds], right_t[pos_inds]) / (torch.max(left_t[pos_inds], right_t[pos_inds]) + 1e-8)) *
                        (torch.min(top_t[pos_inds], bottom_t[pos_inds]) / (torch.max(top_t[pos_inds], bottom_t[pos_inds]) + 1e-8))
                    )

                labels_per_img.append(labels)
                bbox_targets_per_img.append(bbox_targets_assigned)
                ctr_targets_per_img.append(ctr_targets)

            all_labels.append(torch.stack(labels_per_img))
            all_bbox_targets.append(torch.stack(bbox_targets_per_img))
            all_ctr_targets.append(torch.stack(ctr_targets_per_img))

        cls_loss = torch.tensor(0.0, device=cls_logits[0].device)
        reg_loss = torch.tensor(0.0, device=cls_logits[0].device)
        ctr_loss = torch.tensor(0.0, device=cls_logits[0].device)
        num_pos = 0

        for level_idx in range(num_levels):
            cls_logits_per_level = cls_logits[level_idx]
            reg_outputs_per_level = reg_outputs[level_idx]
            ctr_logits_per_level = ctr_logits[level_idx]

            labels_per_level = all_labels[level_idx]
            bbox_targets_per_level = all_bbox_targets[level_idx]
            ctr_targets_per_level = all_ctr_targets[level_idx]

            N, C, H, W = cls_logits_per_level.shape

            cls_logits_flat = cls_logits_per_level.permute(0, 2, 3, 1).reshape(-1, C)
            reg_outputs_flat = reg_outputs_per_level.permute(0, 2, 3, 1).reshape(-1, 4)
            ctr_logits_flat = ctr_logits_per_level.permute(0, 2, 3, 1).reshape(-1)

            labels_flat = labels_per_level.reshape(-1)
            bbox_targets_flat = bbox_targets_per_level.reshape(-1, 4)
            ctr_targets_flat = ctr_targets_per_level.reshape(-1)

            cls_targets = torch.zeros_like(cls_logits_flat)
            pos_mask = labels_flat > 0
            cls_targets[pos_mask, labels_flat[pos_mask] - 1] = 1

            cls_loss += sigmoid_focal_loss(cls_logits_flat, cls_targets, alpha=0.25, gamma=2.0, reduction='sum')

            if pos_mask.sum() > 0:
                points_per_level = points[level_idx].reshape(-1, 2)
                points_flat = points_per_level.unsqueeze(0).expand(N, -1, -1).reshape(-1, 2)

                stride = strides[level_idx]
                reg_targets_pos = bbox_targets_flat[pos_mask] * stride
                reg_outputs_pos = reg_outputs_flat[pos_mask] * stride
                points_pos = points_flat[pos_mask]

                pred_x1 = points_pos[:, 0] - reg_outputs_pos[:, 0]
                pred_y1 = points_pos[:, 1] - reg_outputs_pos[:, 1]
                pred_x2 = points_pos[:, 0] + reg_outputs_pos[:, 2]
                pred_y2 = points_pos[:, 1] + reg_outputs_pos[:, 3]
                pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

                target_x1 = points_pos[:, 0] - reg_targets_pos[:, 0]
                target_y1 = points_pos[:, 1] - reg_targets_pos[:, 1]
                target_x2 = points_pos[:, 0] + reg_targets_pos[:, 2]
                target_y2 = points_pos[:, 1] + reg_targets_pos[:, 3]
                target_boxes = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=1)

                reg_loss += giou_loss(pred_boxes, target_boxes, reduction='sum')

                ctr_logits_pos = ctr_logits_flat[pos_mask]
                ctr_targets_pos = ctr_targets_flat[pos_mask]
                ctr_loss += F.binary_cross_entropy_with_logits(ctr_logits_pos, ctr_targets_pos, reduction='sum')

            num_pos += pos_mask.sum().item()

        num_pos = max(num_pos, 1)

        losses = {
            "cls_loss": cls_loss / num_pos,
            "reg_loss": reg_loss / num_pos,
            "ctr_loss": ctr_loss / num_pos,
            "final_loss": (cls_loss + reg_loss + ctr_loss) / num_pos,
        }

        return losses

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with low object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes and their labels
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep a fixed number of boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        detections = []

        for img_idx, image_shape in enumerate(image_shapes):
            box_cls_per_image = [cls_logits_per_level[img_idx] for cls_logits_per_level in cls_logits]
            box_reg_per_image = [reg_outputs_per_level[img_idx] for reg_outputs_per_level in reg_outputs]
            box_ctr_per_image = [ctr_logits_per_level[img_idx] for ctr_logits_per_level in ctr_logits]

            boxes_all = []
            scores_all = []
            labels_all = []

            for level_idx, (box_cls, box_reg, box_ctr, points_per_level, stride) in enumerate(
                zip(box_cls_per_image, box_reg_per_image, box_ctr_per_image, points, strides)
            ):
                C, H, W = box_cls.shape

                box_cls = box_cls.permute(1, 2, 0).reshape(-1, C).sigmoid()
                box_ctr = box_ctr.permute(1, 2, 0).reshape(-1, 1).sigmoid()
                box_reg = box_reg.permute(1, 2, 0).reshape(-1, 4)

                scores = torch.sqrt(box_cls * box_ctr)

                max_scores, _ = scores.max(dim=1)
                keep_idxs = max_scores > self.score_thresh

                if keep_idxs.sum() == 0:
                    continue

                scores = scores[keep_idxs]
                box_reg = box_reg[keep_idxs]

                points_per_level = points_per_level.reshape(-1, 2)
                points_filtered = points_per_level[keep_idxs]

                topk_idxs = torch.topk(scores.max(dim=1)[0], min(self.topk_candidates, scores.size(0)))[1]
                scores = scores[topk_idxs]
                box_reg = box_reg[topk_idxs]
                points_filtered = points_filtered[topk_idxs]

                x_center = points_filtered[:, 0]
                y_center = points_filtered[:, 1]

                left = box_reg[:, 0] * stride
                top = box_reg[:, 1] * stride
                right = box_reg[:, 2] * stride
                bottom = box_reg[:, 3] * stride

                x1 = x_center - left
                y1 = y_center - top
                x2 = x_center + right
                y2 = y_center + bottom

                boxes = torch.stack([x1, y1, x2, y2], dim=1)

                boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=image_shape[1])
                boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=image_shape[0])

                ws = boxes[:, 2] - boxes[:, 0]
                hs = boxes[:, 3] - boxes[:, 1]
                keep = (ws > 1.0) & (hs > 1.0)

                if keep.sum() == 0:
                    continue

                boxes = boxes[keep]
                scores = scores[keep]

                num_candidates = scores.size(0)
                labels = torch.arange(C, device=scores.device).unsqueeze(0).expand(num_candidates, C)

                boxes = boxes.unsqueeze(1).expand(num_candidates, C, 4).reshape(-1, 4)
                scores = scores.reshape(-1)
                labels = labels.reshape(-1)

                boxes_all.append(boxes)
                scores_all.append(scores)
                labels_all.append(labels)

            if len(boxes_all) == 0:
                detections.append({
                    "boxes": torch.zeros((0, 4), device=cls_logits[0].device),
                    "scores": torch.zeros(0, device=cls_logits[0].device),
                    "labels": torch.zeros(0, dtype=torch.int64, device=cls_logits[0].device),
                })
                continue

            boxes_all = torch.cat(boxes_all, dim=0)
            scores_all = torch.cat(scores_all, dim=0)
            labels_all = torch.cat(labels_all, dim=0)

            keep = batched_nms(boxes_all, scores_all, labels_all, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                "boxes": boxes_all[keep],
                "scores": scores_all[keep],
                "labels": labels_all[keep] + 1,
            })

        return detections
