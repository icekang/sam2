from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image
import torch
import os
from sam2.build_sam import build_sam2_video_predictor
import random
import cv2
import numpy.typing as npt

random.seed(0)
color_pallete = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def maximal_inscribed_circle(binary_label):
    # Convert the label image to a binary image
    binary_label = binary_label.astype(np.uint8)

    dist_map = cv2.distanceTransform(binary_label, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    return center, radius

def get_center_of_maximal_inscribed_circle_in_xy(mask_bool):
    center, _ = maximal_inscribed_circle(mask_bool) # already in x,y as they are the output from opencv
    return np.array([center])

def filter_points_outside_mask(points_in_yx: npt.NDArray[np.int64], mask_bool: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    if not points_in_yx.size:
        return points_in_yx
    
    return points_in_yx[mask_bool[points_in_yx[:, 0], points_in_yx[:, 1]]]

def is_prev_prompt_in_mask(prev_prompt_x: int, prev_prompt_y: int, mask_bool: np.array[bool]) -> bool:
    return mask_bool[prev_prompt_y, prev_prompt_x]


def add_point_prompts(predictor, inference_state, ann_obj_id, frame_index, points, labels):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_index,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

def add_mask_prompt(predictor, inference_state, ann_obj_id, frame_idx, mask_bool):
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=ann_obj_id,
        mask=mask_bool,
    )

def add_prompt(video_label, predictor, inference_state, annotation_every_n):
    video_length = Path(video_label).glob("*.png")
    video_length = len(list(video_length))
    prompts = []
    print('add_prompt/video_length', video_length)

    prev_center_xy = np.array([], shape=(0, 2))
    prev_positive_prompts_yx = np.array([], shape=(0, 2))
    for frame_index in range(0, video_length):
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        mask, palette = load_ann_png(f'{video_label}/{frame_index:05}.png')
        mask_bool = np.all(mask == color_pallete[1].reshape(1, 1, 3), axis=2)
        
        should_add_point_prompt = frame_index % annotation_every_n != 0
        is_first_frame_for_point_prompt = frame_index % annotation_every_n == 1
        if should_add_point_prompt:
            mask_coords = np.argwhere(mask_bool)
            prev_center_xy = filter_points_outside_mask(prev_center_xy[:, ::-1], mask_coords)[:, ::-1]
            prev_positive_prompts_yx = filter_points_outside_mask(prev_positive_prompts_yx, mask_coords)

            no_positive_points_in_mask = len(mask_coords) == 0
            if no_positive_points_in_mask:
                prev_center_xy = np.array([], shape=(0, 2))
                prev_positive_prompts_yx = np.array([], shape=(0, 2))
                continue

            if is_first_frame_for_point_prompt or not prev_center_xy.size:
                prev_center_xy = get_center_of_maximal_inscribed_circle_in_xy(mask_bool)
            else: # if not the first frame, we should add points prompts
                n_sample_points = min(10, len(mask_coords))
                n_positive_prompts = prev_positive_prompts_yx.shape[0]
                positive_prompts_yx = np.array(random.choices(mask_coords, k=n_sample_points - n_positive_prompts))
                positive_prompts_yx = positive_prompts_yx.reshape(-1, 2)
                prev_positive_prompts_yx = np.concatenate([prev_positive_prompts_yx, positive_prompts_yx], axis=0)

            points = np.concatenate([
                prev_center_xy, 
                prev_positive_prompts_yx[:, ::-1] # convert from (y, x) to (x, y)
            ], axis=0)
            labels = np.ones(points.shape[0], dtype=np.int32)
            add_point_prompts(predictor, inference_state, ann_obj_id, frame_index, points, labels)
            prompts.append({'type': 'point', 'frame': frame_index, 'points': prev_center_xy})
        else:
            prev_center_xy = None
            add_mask_prompt(predictor, inference_state, ann_obj_id, frame_index, mask_bool)
            prompts.append({'type': 'mask', 'frame': frame_index, 'mask': mask_bool})
    return prompts

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

def get_label_array(video_dir, video_label):
    video_length = Path(video_dir).glob("*.jpg")
    video_length = len(list(video_length))
    gt = []
    for i in range(0, video_length):
        mask, palette = load_ann_png(f'{video_label}/{i:05}.png')
        mask_bool = np.all(mask == color_pallete[1].reshape(1, 1, 3), axis=2)
        gt.append(mask_bool[None])

    gt = np.concatenate(gt, axis=0)
    gt = gt.astype(np.uint8)
    return gt

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

def run(model_cfg, sam2_checkpoint, output_name):

    dataset_dir = Path('/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/')
    test_image_dir = dataset_dir / 'imagesTs'
    test_label_dir = dataset_dir / 'labelsTs'


    device = torch.device("cuda")

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    resulting_dices_with_prompted_frames = []
    resulting_dices_without_prompted_frames = []

    for volume_path in test_label_dir.glob('*.nii.gz'):
        filename = volume_path.name.replace('.nii.gz', '')

        video_dir = f'/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/imagesTs/{filename}'
        video_label = f'/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/labelsTs/{filename}'
        inference_state = predictor.init_state(video_path=video_dir)

        annotation_every_n=4
        prompts = add_prompt(video_label, predictor, inference_state, annotation_every_n)
        video_segments = run_propagation(predictor, inference_state)
        prediction_output = Path(f'./{output_name}_annotate_every_{annotation_every_n}_consistent_10+1_point_prompts/')
        prediction_output.mkdir(exist_ok=True)
        torch.save(video_segments, prediction_output / f"video_segments_{filename}.pt")
        torch.save(prompts, prediction_output / f"prompts_{filename}.pt")

        pred = convert_video_segments_into_prediction_array(video_segments, object_id=1)
        gt = get_label_array(video_dir, video_label)

        labels, dices, avg_dice = dice_score_of_a_volume(gt, pred)
        print('Dice including prompted frames of', filename, ':', avg_dice)
        resulting_dices_with_prompted_frames.append(avg_dice)

        selector_mask = np.ones(gt.shape[0])
        selector_mask[::annotation_every_n] = 0
        selector_mask = selector_mask.astype(np.bool)
        labels, dices, avg_dice = dice_score_of_a_volume(gt[selector_mask], pred[selector_mask])
        print('Dice excluding prompted frames of ', filename, ':', avg_dice)
        resulting_dices_without_prompted_frames.append(avg_dice)


    print('Average dice including prompted frames:', sum(resulting_dices_with_prompted_frames) / len(resulting_dices_with_prompted_frames))
    print('Average dice excluding prompted frames:', sum(resulting_dices_without_prompted_frames) / len(resulting_dices_without_prompted_frames))
    print('Done')

if __name__ == '__main__':
    run("configs/sam2.1/sam2.1_hiera_s.yaml", "/home/gridsan/nchutisilp/projects/segment-anything-2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_s_MOSE_finetune_scale4_var1.yaml/checkpoints/checkpoint.pt", "sam2.1_hiera_s_MOSE_finetune_scale4_var1.yaml")
    run("configs/sam2.1/sam2.1_hiera_s.yaml", "/home/gridsan/nchutisilp/projects/segment-anything-2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_s_MOSE_finetune_scale4_var2.yaml/checkpoints/checkpoint.pt", "sam2.1_hiera_s_MOSE_finetune_scale4_var2.yaml")
    run("configs/sam2.1/sam2.1_hiera_s.yaml", "/home/gridsan/nchutisilp/projects/segment-anything-2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_s_MOSE_finetune_scale4_var3.yaml/checkpoints/checkpoint.pt", "sam2.1_hiera_s_MOSE_finetune_scale4_var3.yaml")
    run("configs/sam2.1/sam2.1_hiera_s.yaml", "/home/gridsan/nchutisilp/projects/segment-anything-2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_s_MOSE_finetune_scale4_var4.yaml/checkpoints/checkpoint.pt", "sam2.1_hiera_s_MOSE_finetune_scale4_var4.yaml")