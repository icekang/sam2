from pathlib import Path

import numpy as np
from tqdm import tqdm

from helper import color_pallete, load_ann_png
from prompter import Prompter


class VideoPrompter:
    """This class wraps the mask prompter and point prompter to add prompts to a video."""

    def __init__(self, prompters: list[Prompter]):
        self.prompters = prompters

    def add_prompt(
        self,
        video_label: str,
        predictor,
        inference_state,
    ) -> list:
        video_length = Path(video_label).glob("*.png")
        video_length = len(list(video_length))
        prompts = []
        print("add_prompt/video_length", video_length)
        for frame_idx in tqdm(range(0, video_length), desc="Adding prompts"):
            ann_obj_id = 1
            mask, palette = load_ann_png(f"{video_label}/{frame_idx:05}.png")
            mask_bool = np.all(
                mask == color_pallete[1].reshape(1, 1, 3), axis=2
            )

            # Add prompts
            for prompter in self.prompters:
                _prompts = prompter.add_prompt(
                    predictor=predictor,
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    ann_obj_id=ann_obj_id,
                    mask_bool=mask_bool,
                )
                if _prompts:
                    prompts.append(_prompts)
        return prompts
