"""Crop-and-resize preprocessing shared by HMR2 and ViTPose (from hmr4d)."""

import cv2
import numpy as np
import torch

from src.utils.video.reader import read_video_rgb

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=(192, 256)):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    return np.array([w_new, h_new])


def crop_and_resize(img, bbx_xy, bbx_s, dst_size=256, enlarge_ratio=1.2):
    """
    Args:
        img: (H, W, 3)
        bbx_xy: (2,)
        bbx_s: scalar
    """
    hs = bbx_s * enlarge_ratio / 2
    src = np.stack(
        [
            bbx_xy - hs,  # left-up corner
            bbx_xy + np.array([hs, -hs]),  # right-up corner
            bbx_xy,  # center
        ]
    ).astype(np.float32)
    dst = np.array([[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], dtype=np.float32)
    A = cv2.getAffineTransform(src, dst)

    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_xys_final = np.array([*bbx_xy, bbx_s * enlarge_ratio])
    return img_crop, bbx_xys_final


def get_batch(video_path, bbx_xys, img_ds=0.5, img_dst_size=256):
    """Read a video and produce normalized person crops for HMR2 / ViTPose.

    Args:
        video_path: Source video file.
        bbx_xys: (F, 3) tensor of (center_x, center_y, size) in original pixels.
        img_ds: Decode-time downscale factor (speeds up processing).
        img_dst_size: Output crop size.
    Returns:
        imgs: (F, 3, dst, dst) float tensor, ImageNet-normalized RGB.
        bbx_xys: (F, 3) tensor in original pixel scale.
    """
    imgs = read_video_rgb(video_path, max_frames=len(bbx_xys), scale=img_ds)
    if len(imgs) != len(bbx_xys):
        raise ValueError(
            f"Frame count mismatch: video has {len(imgs)} frames, bbx_xys has {len(bbx_xys)}"
        )

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    gt_bbx_size_ds = gt_bbx_size * img_ds
    ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
    imgs = np.stack(
        [
            cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
            for v, d in zip(imgs, ds_factors)
        ]
    )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256)
    return imgs, bbx_xys
