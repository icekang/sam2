import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from sam2.build_sam import build_sam2_video_predictor

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

def dice_score_of_a_volume(gt, pred, ignore_index = 0):
    labels = np.unique(gt)
    dices = []
    for label in labels:
        if label == ignore_index:
            continue
        gt_bool = label == gt
        pred_bool = label == pred
        
        intersection = np.logical_and(gt_bool, pred_bool)
        dices.append(2. * intersection.sum() / (gt_bool.sum() + pred_bool.sum()))

    avg_dice = sum(dices) / len(dices)
    return labels, dices, avg_dice


color_pallete = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def add_prompt(video_label, predictor, inference_state, annotation_every_n):
    video_length = Path(video_label).glob("*.png")
    video_length = len(list(video_length))

    for i in range(0, video_length, annotation_every_n):
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        mask, palette = load_ann_png(f'{video_label}/{i:05}.png')
        mask_bool = np.all(mask == color_pallete[1].reshape(1, 1, 3), axis=2)
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=i,
            obj_id=ann_obj_id,
            mask=mask_bool,
        )

def run_propagation(predictor, inference_state):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
    }
    return video_segments

def convert_video_segments_into_prediction_array(video_segments, object_id=1):
    pred = [video_segments[frame_index][1] for frame_index in range(len(video_segments))]
    pred = np.concatenate(pred, axis=0)
    pred = pred.astype(np.uint8)
    return pred

def get_label_array(video_label):
    video_length = Path(video_label).glob("*.png")
    video_length = len(list(video_length))
    gt = []
    for i in range(0, video_length):
        mask, palette = load_ann_png(f'{video_label}/{i:05}.png')
        mask_bool = np.all(mask == color_pallete[1].reshape(1, 1, 3), axis=2)
        gt.append(mask_bool[None])

    gt = np.concatenate(gt, axis=0)
    gt = gt.astype(np.uint8)
    return gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='SAM2 Calcium OCTv2 Evaluation',
                        description='Evaluate SAM2 with different spacing',
                        epilog='Created with <3 by Naravich Chutisilp')
    parser.add_argument('annotation_every_n', type=int,
                    help='an frenquency of mask prompt in the video')   
    annotation_every_n = parser.parse_args().annotation_every_n
    prediction_output_dir = Path(f'/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/predictions/annotation_every_n_{annotation_every_n}')
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    print('annotation_every_n:', annotation_every_n)
    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    resulting_dices_with_prompted_frames = []
    resulting_dices_without_prompted_frames = []
    result_summary = []
    train_label_dir = Path('/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/labelsTr')
    for volume_path in train_label_dir.glob('*'):
        print(f'Processing {volume_path}')
        filename = volume_path.name

        video_dir = f'/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/imagesTr/{filename}'
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        video_label = f'/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/labelsTr/{filename}'
        inference_state = predictor.init_state(video_path=video_dir)

        add_prompt(video_label, predictor, inference_state, annotation_every_n)
        video_segments = run_propagation(predictor, inference_state)
        torch.save(video_segments, prediction_output_dir / f"video_segments_{filename}.pt")
        pred = convert_video_segments_into_prediction_array(video_segments, object_id=1)
        gt = get_label_array(video_label)

        labels, dices, avg_dice_with_prompted_frames = dice_score_of_a_volume(gt, pred)
        print('Dice including prompted frames of', filename, ':', avg_dice_with_prompted_frames)
        resulting_dices_with_prompted_frames.append(avg_dice_with_prompted_frames)

        selector_mask = np.ones(gt.shape[0])
        selector_mask[::annotation_every_n] = 0
        selector_mask = selector_mask.astype(np.bool)
        labels, dices, avg_dice_without_prompted_frames = dice_score_of_a_volume(gt[selector_mask], pred[selector_mask])
        print('Dice excluding prompted frames of ', filename, ':', avg_dice_without_prompted_frames)
        resulting_dices_without_prompted_frames.append(avg_dice_without_prompted_frames)

        result_summary_item = {
            "filename": filename,
            "dices_with_prompted_frames": avg_dice_with_prompted_frames,
            "dices_without_prompted_frames": avg_dice_without_prompted_frames,
        }
        result_summary.append(result_summary_item)
        with open(prediction_output_dir / "result_summary.json", "w") as f:
            json.dump(result_summary, f)