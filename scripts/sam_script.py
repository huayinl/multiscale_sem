import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import sys
from sam2.build_sam import build_sam2_video_predictor
import cv2
import numpy as np
from tqdm import tqdm

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def setup():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def get_cropped_pixels(img, thresh=254):
    """
    Crop an image to remove black or near-black borders by finding the bounding box
    of pixels above the threshold in any channel.

    Parameters:
    - img: numpy array (H, W, C) - the input image
    - thresh: int (0-255) - threshold below which pixels are considered "black"

    Returns:
    - x0, y0, x1, y1: coordinates of the cropped bounding box, or None if no content found
    """
    # Create mask for pixels that are not black (above thresh in any channel)
    mask = (img > thresh).any(axis=2)
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return x0, y0, x1, y1
    return None
    
def shrink_video(video_dir, save_dir):
    """Shrink video frames by cropping black borders."""
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise NotADirectoryError(f"{save_dir} exists and is not a directory")
    os.makedirs(save_dir, exist_ok=True)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    # gets the overall cropping box across all frames
    min_x0, min_y0 = float('inf'), float('inf')
    max_x1, max_y1 = -1, -1
    for frame_name in tqdm(frame_names, desc="Calculating cropping box"):
        img_fp = os.path.join(video_dir, frame_name)
        img = cv2.imread(img_fp)  # BGR
        x0, y0, x1, y1 = get_cropped_pixels(img, thresh=254)
        if x0 is not None:
            min_x0 = min(min_x0, x0)
            min_y0 = min(min_y0, y0)
            max_x1 = max(max_x1, x1)
            max_y1 = max(max_y1, y1)
        # print(f"Frame {frame_name} | x0 {x0} (min {min_x0}) | y0 {y0} (min {min_y0}) | x1 {x1} (max {max_x1}) | y1 {y1} (max {max_y1})", end="\r")
    
    # crop all frames using the overall cropping box
    for i, frame_name in tqdm(enumerate(frame_names), desc="Cropping and saving frames"):
        img_fp = os.path.join(video_dir, frame_name)
        save_fp = os.path.join(save_dir, f"{i}.jpg")
        img = cv2.imread(img_fp)  # BGR
        cropped_img = img[min_y0:max_y1, min_x0:max_x1]
        cv2.imwrite(save_fp, cropped_img)

print("Setting up device...")
device = setup()
sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
print(f"Adding SAM 2 Video Predictor...")
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


SHRINK_VIDEO = False
video_dir = "small_thumbnails_cropped"

# (1) Crop video frames to remove black borders (first time only)
# So we're working with smaller videos, faster inference, and better segmentation results
if SHRINK_VIDEO:
    shrink_video(video_dir, f"{video_dir}_cropped")
    video_dir = f"{video_dir}_cropped"

# (2) Run video segmentation
inference_state = predictor.init_state(video_path=video_dir)
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

# add prompts
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[1300, 3000], [1400, 2700]], dtype=np.float32)
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# run propagation throughout the video and collect the results in a dict
results_dir = f"{video_dir}_masks"
os.makedirs(results_dir, exist_ok=True)
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    save_path = os.path.join(results_dir, f"{out_frame_idx}.jpg")
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_uint8[0])

