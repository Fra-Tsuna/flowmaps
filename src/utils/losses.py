import torch

def _inter_union_3d(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute intersection and union volumes for 3D axis-aligned boxes.

    boxes1, boxes2: (..., 6) in (x1, y1, z1, x2, y2, z2) format
    Returns:
        inter: (...,)
        union: (...,)
    """
    x1, y1, z1, x2, y2, z2 = boxes1.unbind(dim=-1)
    x1g, y1g, z1g, x2g, y2g, z2g = boxes2.unbind(dim=-1)
    
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"
    assert (z2 >= z1).all(), "bad box: z1 larger than z2"

    # Intersection box
    ix1 = torch.maximum(x1, x1g)
    iy1 = torch.maximum(y1, y1g)
    iz1 = torch.maximum(z1, z1g)
    ix2 = torch.minimum(x2, x2g)
    iy2 = torch.minimum(y2, y2g)
    iz2 = torch.minimum(z2, z2g)

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    id_ = (iz2 - iz1).clamp(min=0)

    inter = iw * ih * id_

    # Volumes
    w1 = (x2 - x1).clamp(min=0)
    h1 = (y2 - y1).clamp(min=0)
    d1 = (z2 - z1).clamp(min=0)
    vol1 = w1 * h1 * d1

    w2 = (x2g - x1g).clamp(min=0)
    h2 = (y2g - y1g).clamp(min=0)
    d2 = (z2g - z1g).clamp(min=0)
    vol2 = w2 * h2 * d2

    union = vol1 + vol2 - inter
    return inter, union


def _diou_iou_loss_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
):
    """
    3D analogue of torchvision.ops.diou_loss._diou_iou_loss for boxes in
    (x1, y1, z1, x2, y2, z2) format.
    Returns:
        loss_diou: (...,)
        iou: (...,)
    """
    intsct, union = _inter_union_3d(boxes1, boxes2)
    iou = intsct / (union + eps)

    x1, y1, z1, x2, y2, z2 = boxes1.unbind(dim=-1)
    x1g, y1g, z1g, x2g, y2g, z2g = boxes2.unbind(dim=-1)

    # Smallest enclosing box in 3D
    xc1 = torch.minimum(x1, x1g)
    yc1 = torch.minimum(y1, y1g)
    zc1 = torch.minimum(z1, z1g)
    xc2 = torch.maximum(x2, x2g)
    yc2 = torch.maximum(y2, y2g)
    zc2 = torch.maximum(z2, z2g)

    # Diagonal distance squared of enclosing box
    diagonal_distance_squared = (
        (xc2 - xc1) ** 2 +
        (yc2 - yc1) ** 2 +
        (zc2 - zc1) ** 2 +
        eps
    )

    # Centers of boxes
    x_p = (x1 + x2) / 2.0
    y_p = (y1 + y2) / 2.0
    z_p = (z1 + z2) / 2.0

    x_g = (x1g + x2g) / 2.0
    y_g = (y1g + y2g) / 2.0
    z_g = (z1g + z2g) / 2.0

    centers_distance_squared = (
        (x_p - x_g) ** 2 +
        (y_p - y_g) ** 2 +
        (z_p - z_g) ** 2
    )

    # 3D DIoU loss
    loss = 1.0 - iou + centers_distance_squared / diagonal_distance_squared
    return loss, iou


def complete_box_iou_loss_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    3D version of torchvision.ops.complete_box_iou_loss for axis-aligned boxes.

    Args:
        boxes1: (..., 6) or (6,) in (x1, y1, z1, x2, y2, z2) format
        boxes2: same shape as boxes1
        reduction: "none" | "mean" | "sum"
        eps: small constant for numerical stability
    """
    if not boxes1.is_floating_point():
        boxes1 = boxes1.float()
    if not boxes2.is_floating_point():
        boxes2 = boxes2.float()

    diou_loss, iou = _diou_iou_loss_3d(boxes1, boxes2, eps=eps)

    x1, y1, z1, x2, y2, z2 = boxes1.unbind(dim=-1)
    x1g, y1g, z1g, x2g, y2g, z2g = boxes2.unbind(dim=-1)

    # Side lengths (you said boxes are valid, but clamp keeps log/ratios safe)
    w_pred = (x2 - x1).clamp(min=eps)
    h_pred = (y2 - y1).clamp(min=eps)
    d_pred = (z2 - z1).clamp(min=eps)

    w_gt = (x2g - x1g).clamp(min=eps)
    h_gt = (y2g - y1g).clamp(min=eps)
    d_gt = (z2g - z1g).clamp(min=eps)

    # Log-ratio aspect penalty on the 3 independent 3D aspect ratios
    # different from original v-value computation of torchvision
    r1_pred = torch.log(w_pred / h_pred)
    r2_pred = torch.log(w_pred / d_pred)
    r3_pred = torch.log(h_pred / d_pred)

    r1_gt = torch.log(w_gt / h_gt)
    r2_gt = torch.log(w_gt / d_gt)
    r3_gt = torch.log(h_gt / d_gt)

    v = ((r1_pred - r1_gt) ** 2 + (r2_pred - r2_gt) ** 2 + (r3_pred - r3_gt) ** 2) / 3.0

    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    loss = diou_loss + alpha * v

    def _reduce(t):
        if reduction == "none":
            return t
        elif reduction == "mean":
            return t.mean() if t.numel() > 0 else 0.0 * t.sum()
        elif reduction == "sum":
            return t.sum()
        else:
            raise ValueError(f"Invalid reduction '{reduction}'. Supported: 'none', 'mean', 'sum'.")

    iou_loss = 1.0 - iou
    centers_distance_squared = diou_loss - iou_loss  # diou = (1-iou) + dist/diag
    return {
        "loss": _reduce(loss),
        "iou_loss": _reduce(iou_loss),
        "distance_loss": _reduce(centers_distance_squared),
        "aspect_loss": _reduce(alpha * v),
    }
