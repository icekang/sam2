import argparse
from pathlib import Path

import torch
from helper import (
    calcuate_dice_score,
    get_negative_video_label,
    get_splits,
    get_video_dir,
    get_video_label,
    run_propagation,
)
from prompter import (
    KBorderPointsPrompter,
    KBorderPointsPrompterV2,
    KBorderPointsPrompterV3,
    KConsistentPointPrompter,
    KNegativeConsistentPointsPrompter,
    MaskPrompter,
    RandomPointPrompter,
)
from video_prompter import VideoPrompter

from sam2.build_sam import build_sam2_video_predictor

EVERY_N = 16


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
        neg_video_label = get_negative_video_label(filename=filename)
        print("Processing", filename)
        print(video_dir)
        inference_state = predictor.init_state(video_path=video_dir)

        # Add mask every EVERY_N frame
        prompts = video_prompter.add_prompt(
            video_label=video_label,
            neg_video_label=neg_video_label,
            predictor=predictor,
            inference_state=inference_state,
        )

        # Run inference
        video_segments = run_propagation(predictor, inference_state)

        # Save the results for later visualization
        prediction_output = Path(f"./{output_name}_annotate_every_{EVERY_N}/")
        prediction_output.mkdir(exist_ok=True)
        torch.save(
            video_segments, prediction_output / f"video_segments_{filename}.pt"
        )
        torch.save(prompts, prediction_output / f"prompts_{filename}.pt")

        # Calculate Dice score
        dice_with_masks, dice_without_masks = calcuate_dice_score(
            video_segments=video_segments,
            filename=filename,
            annotation_every_n=EVERY_N,
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

def setup_argument_parser():
    """
    Set up an argument parser with flexible options for different prompters.

    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Evaluate video segmentation model."
    )

    # Base required arguments
    parser.add_argument(
        "--prompter_names",
        nargs="+",
        required=True,
        help="List of prompter names to use",
        choices=[
            "mask", "random_point", "consistent_point",
            "k_consistent_point", "k_neg_consistent_point",
            "k_border", "k_border_2", "k_border_3"
        ]
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold number for cross-validation"
    )
    parser.add_argument(
        "--mask_every_n",
        type=int,
        required=True,
        help="Prompt mask every n frames, and prompt points in between"
    )

    # Optional arguments for specific prompters
    parser.add_argument(
        "--k_consistent_point_k",
        type=int,
        default=0,
        help="K value for k_consistent_point prompter"
    )
    parser.add_argument(
        "--k_neg_consistent_point_k",
        type=int,
        default=19,
        help="K value for k_neg_consistent_point prompter"
    )
    parser.add_argument(
        "--k_border_pos_k",
        type=int,
        default=9,
        help="Positive K value for k_border prompter"
    )
    parser.add_argument(
        "--k_border_neg_k",
        type=int,
        default=19,
        help="Negative K value for k_border prompter"
    )
    parser.add_argument(
        "--k_border_2_pos_k",
        type=int,
        default=9,
        help="Positive K value for k_border_2 prompter"
    )
    parser.add_argument(
        "--k_border_2_neg_k",
        type=int,
        default=9,
        help="Negative K value for k_border_2 prompter"
    )
    parser.add_argument(
        "--k_border_3_pos_k",
        type=int,
        default=9,
        help="Positive K value for k_border_3 prompter"
    )
    parser.add_argument(
        "--k_border_3_neg_k",
        type=int,
        default=9,
        help="Negative K value for k_border_3 prompter"
    )

    return parser

def parse_prompter_args(prompter_name, args):
    """
    Parse arguments specific to each prompter type.

    Args:
        prompter_name (str): Name of the prompter
        args (Namespace): Parsed command-line arguments

    Returns:
        dict: Keyword arguments for the specific prompter
    """
    prompter_kwargs = {
        'annotation_every_n': args.mask_every_n
    }

    # Extract custom arguments for specific prompters
    if prompter_name == 'k_consistent_point':
        prompter_kwargs['k'] = getattr(args, f'{prompter_name}_k', 0)

    elif prompter_name == 'k_neg_consistent_point':
        prompter_kwargs['k'] = getattr(args, f'{prompter_name}_k', 19)

    elif prompter_name == 'k_border':
        prompter_kwargs['pos_k'] = getattr(args, f'{prompter_name}_pos_k', 9)
        prompter_kwargs['neg_k'] = getattr(args, f'{prompter_name}_neg_k', 19)

    elif prompter_name == 'k_border_2':
        prompter_kwargs['pos_k'] = getattr(args, f'{prompter_name}_pos_k', 9)
        prompter_kwargs['neg_k'] = getattr(args, f'{prompter_name}_neg_k', 9)

    elif prompter_name == 'k_border_3':
        prompter_kwargs['pos_k'] = getattr(args, f'{prompter_name}_pos_k', 9)
        prompter_kwargs['neg_k'] = getattr(args, f'{prompter_name}_neg_k', 9)

    return prompter_kwargs

def get_video_prompter(*prompter_names: list[str], args=None):
    """
    Get a VideoPrompter object with the specified prompters.

    Args:
        prompter_names (list): List of prompter names
        args (Namespace, optional): Parsed command-line arguments

    Returns:
        VideoPrompter: Configured video prompter
    """
    prompters = []
    for prompter_name in prompter_names:
        match prompter_name:
            case "mask":
                prompters.append(MaskPrompter(annotation_every_n=args.mask_every_n))
            case "random_point":
                prompters.append(RandomPointPrompter(annotation_every_n=args.mask_every_n))
            case "consistent_point":
                prompters.append(ConsistentPointPrompter(annotation_every_n=args.mask_every_n))
            case "k_consistent_point":
                kwargs = parse_prompter_args(prompter_name, args)
                prompters.append(KConsistentPointPrompter(**kwargs))
            case "k_neg_consistent_point":
                kwargs = parse_prompter_args(prompter_name, args)
                prompters.append(KNegativeConsistentPointsPrompter(**kwargs))
            case "k_border":
                kwargs = parse_prompter_args(prompter_name, args)
                prompters.append(KBorderPointsPrompter(**kwargs))
            case "k_border_2":
                kwargs = parse_prompter_args(prompter_name, args)
                prompters.append(KBorderPointsPrompterV2(**kwargs))
            case "k_border_3":
                kwargs = parse_prompter_args(prompter_name, args)
                prompters.append(KBorderPointsPrompterV3(**kwargs))
            case _:
                raise ValueError(f"Unknown prompter name: {prompter_name}")

    return VideoPrompter(prompters=prompters)

# Create detailed output name with argument values
def get_prompter_arg_string(prompter_name, args):
    """Generate a string of argument values for a specific prompter."""
    if prompter_name == 'k_consistent_point':
        return f"k{getattr(args, f'{prompter_name}_k', 0)}"
    elif prompter_name == 'k_neg_consistent_point':
        return f"k{getattr(args, f'{prompter_name}_k', 19)}"
    elif prompter_name in ['k_border', 'k_border_2', 'k_border_3']:
        pos_k = getattr(args, f'{prompter_name}_pos_k', 9)
        neg_k = getattr(args, f'{prompter_name}_neg_k', 9)
        return f"pos{pos_k}_neg{neg_k}"
    return ""

if __name__ == "__main__":
    # Set up argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Get project root directory more reliably
    project_root = Path(__file__).resolve().parent.parent

    # Create video prompter with flexible arguments
    video_prompter = get_video_prompter(*args.prompter_names, args=args)

    # Generate detailed strategy name with argument values
    detailed_strategy_name = "::".join([
        f"{name}_{get_prompter_arg_string(name, args)}"
        if name in ['k_consistent_point', 'k_neg_consistent_point', 'k_border', 'k_border_2', 'k_border_3']
        else name
        for name in sorted(args.prompter_names)
    ])

    # Rest of the script remains the same
    model_cfg_name = f"fold{args.fold}.yaml"

    run(
        model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
        sam2_checkpoint=str(
            project_root
            / "sam2_logs/configs/sam2.1_training"
            / "splits_final"
            / model_cfg_name
            / "checkpoints/checkpoint.pt"
        ),
        output_name=f"{model_cfg_name[:-5]}_{detailed_strategy_name}",
        fold=args.fold,
        video_prompter=video_prompter,
    )