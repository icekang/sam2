import argparse
from functools import partial
from pathlib import Path
from typing import Callable
from warnings import deprecated
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
from helper import get_splits, get_video_label, get_video_dir
from evaluate import get_prompter_arg_string, setup_argument_parser

color_pallete = np.array([[0, 0, 0], [0, 255, 0], [0, 255, 0]])

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

def show_points(coords, labels, ax: plt.axes, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    return ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def update_animation_full(
    ax: plt.Axes,
    predictions: dict,
    frame_idx2prompt: dict,
    image_paths: list[Path],
    groundtruth_paths: list[Path],
    EVERY_N: int,
    frame: int,
) -> tuple[plt.axes, plt.axes, plt.axes]:
    ax[2].clear()
    ax[0].set_title(f"Image ({frame} | {image_paths[frame].name})")

    if frame % EVERY_N == 0:
        ax[2].set_title("SAM2 Prediction (mask prompted)")
    else:
        ax[2].set_title("SAM2 Prediction")

    image = Image.open(image_paths[frame])
    image = image.convert("RGBA")

    gt = Image.open(groundtruth_paths[frame])
    gt = gt.convert("RGBA")

    prediction = Image.fromarray(
        color_pallete[predictions[frame][1][0].astype(np.uint8)].astype(
            np.uint8
        )
    )
    prediction = prediction.convert("RGBA")

    new_image = Image.blend(gt, prediction, 0.5)
    new_image = Image.blend(image, new_image, 0.5)

    ax_2_output = ax[2].imshow(prediction)
    if frame in frame_idx2prompt:
        prompts = frame_idx2prompt[frame]
        draw_prompts(ax, prompts)
    return ax[0].imshow(new_image), ax[1].imshow(gt), ax_2_output


def draw_prompts(ax: plt.axes, prompts: list[dict]):
    """
    Draw prompts on the axes.
    """
    for prompt in prompts:
        prompt_type = prompt["type"]
        if prompt_type == "mask":
            show_mask(
                mask=prompt["mask"],
                ax=ax[2],
                obj_id=prompt["obj_id"],
                random_color=False,
            )
        elif prompt_type == "point":
            if "neg_points" in prompt:
                neg_points_xy = prompt["neg_points"]
                N, two = neg_points_xy.shape
                show_points(
                    coords=neg_points_xy,
                    labels=np.array([0] * N),
                    ax=ax[2],
                    marker_size=200,
                )
            if "points" in prompt:
                points_xy = prompt["points"]
                N, two = points_xy.shape
                ax_2_output = show_points(
                    coords=points_xy,
                    labels=np.array([1] * N),
                    ax=ax[2],
                    marker_size=200,
                )


def get_update_animation(
    ax: plt.Axes,
    predictions: dict,
    frame_idx2prompt: dict,
    image_paths: list[Path],
    groundtruth_paths: list[Path],
    EVERY_N: int,
) -> Callable:
    # Partially apply the arguments except for frame, using python's functools.partial
    return partial(
        update_animation_full,
        ax,
        predictions,
        frame_idx2prompt,
        image_paths,
        groundtruth_paths,
        EVERY_N,
    )


def get_patients(fold: int):
    splits, _, _ = get_splits()
    patients = splits[fold]["val"]
    return patients


def create_videos(fold: int, prediction_output: Path, EVERY_N: int):
    patients = get_patients(fold)

    for patient_id in patients:
        prediction_path = prediction_output / f"video_segments_{patient_id}.pt"
        predictions = torch.load(prediction_path, weights_only=False)

        prompt_path = prediction_output / f"prompts_{patient_id}.pt"
        prompts = torch.load(prompt_path, weights_only=False)
        frame_idx2prompt = aggregate_prompts(prompts)

        groundtruth_path = Path(get_video_label(patient_id))
        image_path = Path(get_video_dir(patient_id))

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        image_paths = list(image_path.glob("*.jpg"))
        image_paths.sort()

        groundtruth_paths = list(groundtruth_path.glob("*.png"))
        groundtruth_paths.sort()

        ax[1].set_title("Groundtruth")

        update_animation = get_update_animation(
            ax=ax,
            predictions=predictions,
            frame_idx2prompt=frame_idx2prompt,
            image_paths=image_paths,
            groundtruth_paths=groundtruth_paths,
            EVERY_N=EVERY_N,
        )

        update_animation(0)
        print(prediction_output, patient_id)
        ani = animation.FuncAnimation(
            fig=fig, func=update_animation, frames=len(image_paths)
        )
        ani.save(
            prediction_output / f"{patient_id}.mp4",
            progress_callback=lambda i, n: print("\r", i, "/", n, end=""),
        )


@deprecated("Will be removed in the future when the prompter is fixed.")
def _prompt_list_to_dict(prompts: list[dict]) -> dict:
    """
    Convert a list of prompts to a dictionary.
    """
    if isinstance(prompts, dict):
        return prompts

    if len(prompts) != 1:
        raise ValueError("Only one prompt per frame is supported.")

    prompt = prompts[0]
    return prompt


def aggregate_prompts(prompts: list[dict]) -> dict:
    """
    Aggregate prompts into a single dictionary.
    """
    aggregated_prompts = {}
    for prompt in prompts:
        # prompt = _prompt_list_to_dict(prompt)
        frame = prompt["frame"]
        if frame not in aggregated_prompts:
            aggregated_prompts[frame] = []
        aggregated_prompts[frame].append(prompt)
    return aggregated_prompts


def get_prediction_output(fold: int, strategy_name: str, EVERY_N: int) -> Path:
    """
    Get the prediction output name.
    """
    return Path(f"fold{fold}_{strategy_name}_annotate_every_{EVERY_N}")

def get_show_annoation_function(
    fold: int, prediction_output: Path, patient_idx: int, EVERY_N: int
) -> tuple[Callable, plt.Figure, plt.Axes]:
    """
    Returns a function that can be used to visualize the annotation of a patient."
    """
    # Get the patient id
    patients = get_patients(fold)
    patient_id = patients[patient_idx]

    # Load the predictions and prompts
    prediction_path = prediction_output / f"video_segments_{patient_id}.pt"
    predictions = torch.load(prediction_path, weights_only=False)

    # Load the prompts
    prompt_path = prediction_output / f"prompts_{patient_id}.pt"
    prompts = torch.load(prompt_path, weights_only=False)

    # Create a dictionary that maps frame index to prompt
    frame_idx2prompt = aggregate_prompts(prompts)

    # Load the groundtruth and image paths
    groundtruth_path = Path(get_video_label(patient_id))
    image_path = Path(get_video_dir(patient_id))

    # Create a figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    # Load the image and groundtruth paths
    image_paths = list(image_path.glob("*.jpg"))
    image_paths.sort()

    groundtruth_paths = list(groundtruth_path.glob("*.png"))
    groundtruth_paths.sort()

    ax[1].set_title("Groundtruth")

    update_animation = get_update_animation(
        ax=ax,
        predictions=predictions,
        frame_idx2prompt=frame_idx2prompt,
        image_paths=image_paths,
        groundtruth_paths=groundtruth_paths,
        EVERY_N=EVERY_N,
    )
    return update_animation, fig, ax


if __name__ == "__main__":
    # Set up argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Get project root directory more reliably
    project_root = Path(__file__).resolve().parent.parent

    # Generate detailed strategy name with argument values
    detailed_strategy_name = "::".join([
        f"{name}_{get_prompter_arg_string(name, args)}"
        if name in ['k_consistent_point', 'k_neg_consistent_point', 'k_border', 'k_border_2', 'k_border_3']
        else name
        for name in sorted(args.prompter_names)
    ])

    prediction_output_name = get_prediction_output(
        fold=args.fold, strategy_name=detailed_strategy_name, EVERY_N=args.mask_every_n
    )
    prediction_output = Path(prediction_output_name)

    create_videos(args.fold, prediction_output, args.mask_every_n)
