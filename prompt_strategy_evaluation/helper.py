import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor
from training.utils.train_utils import register_omegaconf_resolvers

random.seed(0)
color_pallete = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette




def add_prompt(video_label: str, predictor, inference_state, annotation_every_n: int):
    video_length = Path(video_label).glob("*.png")
    video_length = len(list(video_length))
    prompts = []
    print("add_prompt/video_length", video_length)
    for i in range(0, video_length):
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        mask, palette = load_ann_png(f"{video_label}/{i:05}.png")
        mask_bool = np.all(mask == color_pallete[1].reshape(1, 1, 3), axis=2)

        if i % annotation_every_n != 0:
            random_point = add_random_positive_point(
                predictor=predictor,
                inference_state=inference_state,
                frame_idx=i,
                ann_obj_id=ann_obj_id,
                mask_bool=mask_bool,
            )
            if random_point:
                prompts.append({"type": "point", "frame": i, "points": random_point})
        else:
            add_mask_prompt(
                predictor=predictor,
                inference_state=inference_state,
                frame_idx=i,
                ann_obj_id=ann_obj_id,
                mask_bool=mask_bool,
            )
            prompts.append({"type": "mask", "frame": i, "mask": mask_bool})
    return prompts


def add_random_positive_point(
    predictor, inference_state, frame_idx: int, ann_obj_id: int, mask_bool: npt.NDArray
):
    mask_coords = np.argwhere(mask_bool)
    if len(mask_coords) == 0:  # No label in this frame
        return None
    random_point = random.choice(mask_coords)
    random_point = random_point[::-1]  # (y, x) -> (x, y)
    random_point = np.array([random_point])
    labels = np.array(
        [1], np.int32
    )  # for labels, `1` means positive click and `0` means negative click
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=ann_obj_id,
        points=random_point,
        labels=labels,
    )
    return random_point


def add_mask_prompt(
    predictor, inference_state, frame_idx: int, ann_obj_id: int, mask_bool: npt.NDArray
):
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=ann_obj_id,
        mask=mask_bool,
    )


def run_propagation(predictor, inference_state):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


def convert_video_segments_into_prediction_array(video_segments, object_id=1):
    pred = [
        video_segments[frame_index][1] for frame_index in range(len(video_segments))
    ]
    pred = np.concatenate(pred, axis=0)
    pred = pred.astype(np.uint8)
    return pred


def get_label_array(video_dir, video_label):
    video_length = Path(video_dir).glob("*.jpg")
    video_length = len(list(video_length))
    gt = []
    for i in range(0, video_length):
        mask, palette = load_ann_png(f"{video_label}/{i:05}.png")
        mask_bool = np.all(mask == color_pallete[1].reshape(1, 1, 3), axis=2)
        gt.append(mask_bool[None])

    gt = np.concatenate(gt, axis=0)
    gt = gt.astype(np.uint8)
    return gt


def dice_score_of_a_volume(gt, pred, ignore_index=0):
    labels = np.unique(gt)
    dices = []
    for label in labels:
        if label == ignore_index:
            continue
        gt_bool = label == gt
        pred_bool = label == pred

        intersection = np.logical_and(gt_bool, pred_bool)
        dices.append(2.0 * intersection.sum() / (gt_bool.sum() + pred_bool.sum()))

    avg_dice = sum(dices) / len(dices)
    return labels, dices, avg_dice


def get_splits():
    DATASET_ID = "307"
    DATASET_NAME = "Sohee_Calcium_OCT_CrossValidation"

    nnUNet_preprocessed = Path(os.environ["nnUNet_preprocessed"])
    nnUNet_raw = Path(os.environ["nnUNet_raw"])

    split_filepath = (
        nnUNet_preprocessed / f"Dataset{DATASET_ID}_{DATASET_NAME}/splits_final.json"
    )
    imageTr_dir = nnUNet_raw / f"Dataset{DATASET_ID}_{DATASET_NAME}/imagesTr"
    labelTr_dir = nnUNet_raw / f"Dataset{DATASET_ID}_{DATASET_NAME}/labelsTr"

    with open(split_filepath, "r") as f:
        splits = json.load(f)

    return splits, imageTr_dir, labelTr_dir


def get_video_dir(filename: str):
    video_dir = f"/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/imagesTr/{filename}"
    return video_dir


def get_video_label(filename: str):
    video_label = f"/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/labelsTr/{filename}"
    return video_label


def calcuate_dice_score(
    video_segments: dict, filename: str, annotation_every_n: int
) -> tuple[float, float]:
    video_dir = get_video_dir(filename=filename)
    video_label = get_video_label(filename=filename)

    pred = convert_video_segments_into_prediction_array(video_segments, object_id=1)
    gt = get_label_array(video_dir, video_label)

    labels, dices, avg_dice_with_prompt_masks = dice_score_of_a_volume(gt, pred)
    print(
        "Dice including prompted frames of", filename, ":", avg_dice_with_prompt_masks
    )

    selector_mask = np.ones(gt.shape[0])
    selector_mask[::annotation_every_n] = 0
    selector_mask = selector_mask.astype(np.bool)
    labels, dices, avg_dice_without_prompted_masks = dice_score_of_a_volume(
        gt[selector_mask], pred[selector_mask]
    )
    print(
        "Dice excluding prompted frames of ",
        filename,
        ":",
        avg_dice_without_prompted_masks,
    )
    return avg_dice_with_prompt_masks, avg_dice_without_prompted_masks


def run(model_cfg: str, sam2_checkpoint: str, output_name: str, fold: int):
    device = torch.device("cpu")

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    resulting_dices_with_prompted_frames = []
    resulting_dices_without_prompted_frames = []

    # Get splits to get the image filenames in the validation set for that split.
    splits, test_image_dir, test_label_dir = get_splits()
    print(splits)
    split = splits[fold]
    print(fold)
    val_ids = split["val"]
    for filename in val_ids:
        # Load the image
        video_dir = get_video_dir(filename=filename)
        video_label = get_video_label(filename=filename)
        print("Processing", filename)
        print(video_dir)
        inference_state = predictor.init_state(video_path=video_dir)

        # Add mask every 4 frame
        annotation_every_n = 4
        prompts = add_prompt(
            video_label, predictor, inference_state, annotation_every_n
        )

        # Run inference
        video_segments = run_propagation(predictor, inference_state)

        # Save the results for later visualization
        prediction_output = Path(
            f"./{output_name}_annotate_every_{annotation_every_n}/"
        )
        prediction_output.mkdir(exist_ok=True)
        torch.save(video_segments, prediction_output / f"video_segments_{filename}.pt")
        torch.save(prompts, prediction_output / f"prompts_{filename}.pt")

        # Calculate Dice score
        dice_with_masks, dice_without_masks = calcuate_dice_score(
            video_segments=video_segments,
            filename=filename,
            annotation_every_n=annotation_every_n,
        )
        resulting_dices_with_prompted_frames.append(dice_with_masks)
        resulting_dices_without_prompted_frames.append(
            resulting_dices_without_prompted_frames
        )

    print(
        "Average dice including prompted frames:",
        sum(resulting_dices_with_prompted_frames)
        / len(resulting_dices_with_prompted_frames),
    )
    print(
        "Average dice excluding prompted frames:",
        sum(resulting_dices_without_prompted_frames)
        / len(resulting_dices_without_prompted_frames),
    )
    print("Done")
