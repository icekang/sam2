from pathlib import Path

import torch
from prompt_strategy_evaluation.video_prompter import VideoPrompter
from prompter import MaskPrompter, Prompter

from prompt_strategy_evaluation.helper import (
    calcuate_dice_score,
    get_splits,
    get_video_dir,
    get_video_label,
    run_propagation,
)
from sam2.build_sam import build_sam2_video_predictor
import argparse


def run(
    model_cfg: str,
    sam2_checkpoint: str,
    output_name: str,
    fold: int,
    video_prompter: VideoPrompter,
):
    device = (
        torch.device("cpu")
        if not torch.cuda.is_available()
        else torch.device("cuda")
    )

    predictor = build_sam2_video_predictor(
        model_cfg, sam2_checkpoint, device=device
    )

    resulting_dices_with_prompted_frames = []
    resulting_dices_without_prompted_frames = []

    # Get splits to get the image filenames in the validation set for that split.
    splits, test_image_dir, test_label_dir = get_splits()
    split = splits[fold]
    print(f"Fold {fold} with split {split}")

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
        prompts = video_prompter.add_prompt(
            video_label, predictor, inference_state, annotation_every_n
        )

        # Run inference
        video_segments = run_propagation(predictor, inference_state)

        # Save the results for later visualization
        prediction_output = Path(
            f"./{output_name}_annotate_every_{annotation_every_n}/"
        )
        prediction_output.mkdir(exist_ok=True)
        torch.save(
            video_segments, prediction_output / f"video_segments_{filename}.pt"
        )
        torch.save(prompts, prediction_output / f"prompts_{filename}.pt")

        # Calculate Dice score
        dice_with_masks, dice_without_masks = calcuate_dice_score(
            video_segments=video_segments,
            filename=filename,
            annotation_every_n=annotation_every_n,
        )
        resulting_dices_with_prompted_frames.append(dice_with_masks)
        resulting_dices_without_prompted_frames.append(dice_without_masks)

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


def get_video_prompter(*prompter_names: list[str]):
    """Get a VideoPrompter object with the specified prompters."""
    prompters = []
    for prompter_name in prompter_names:
        match prompter_name:
            case "mask":
                prompters.append(MaskPrompter())
            case "random_point":
                prompters.append(RandomPointPrompter())
            case "consistent_point":
                prompters.append(ConsistentPointPrompter())
            case "k_consistent_point":
                prompters.append(KConsistentPointPrompter())
            case _:
                raise ValueError(f"Unknown prompter name: {prompter_name}")

    return VideoPrompter(prompters=prompters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate video segmentation model."
    )
    parser.add_argument(
        "--prompter_names",
        nargs="+",
        required=True,
        help="List of prompter names to use (e.g., mask, random_point, consistent_point, k_consistent_point)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold number for cross-validation",
    )
    parser.add_argument(
        "--model_cfg_name",
        type=str,
        default="sam2.1_hiera_s_MOSE_finetune.yaml",
        help="Model configuration filename",
    )
    args = parser.parse_args()

    # Get project root directory more reliably
    project_root = Path(__file__).resolve().parent.parent

    video_prompter = get_video_prompter(*args.prompter_names)

    # Get project root directory more reliably
    project_root = Path(__file__).resolve().parent.parent

    video_prompter = get_video_prompter(*args.prompter_names)
    model_cfg_name = args.model_cfg_name
    run(
        model_cfg=str(project_root / "configs/sam2.1/sam2.1_hiera_s.yaml"),
        sam2_checkpoint=str(
            project_root
            / "sam2_logs/configs/sam2.1_training"
            / model_cfg_name
            / "checkpoints/checkpoint.pt"
        ),
        output_name=f"{model_cfg_name[:-5]}",
        fold=args.fold,
        video_prompter=video_prompter,
    )
