import random
import cv2
import numpy as np
import numpy.typing as npt
from abc import abstractmethod


class Prompter:
    def __init__(
        self,
        annotation_every_n: int,
    ):
        self.annotation_every_n = annotation_every_n

    @abstractmethod
    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
    ):
        pass


class ConsistentPointPrompter:
    """A prompter that put the same point from the previous frame."""

    def __init__(
        self,
        annotation_every_n: int,
    ):
        self.prev_center = None
        self.annotation_every_n = annotation_every_n

    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
    ):
        mask_coords = np.argwhere(mask_bool)
        if len(mask_coords) == 0:
            self.prev_center = None
            return None
        if (
            frame_idx % self.annotation_every_n == 1 or self.prev_center is None
        ):  # The frame we should add a new consistent point
            center, radius = maximal_inscribed_circle(
                mask_bool
            )  # already in x,y as they are the output from opencv
            random_point = np.array([center])
            self.prev_center = random_point
        assert (
            self.prev_center is not None
        ), f"prev_center should not be None at frame {frame_idx} while annotation is every {self.annotation_every_n} (modulo is {frame_idx % annotation_every_n})"
        labels = np.array(
            [1], np.int32
        )  # for labels, `1` means positive click and `0` means negative click
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=ann_obj_id,
            points=self.prev_center,
            labels=labels,
        )
        return {'type': 'point', 'frame': frame_idx, 'points': self.prev_center}
        


class RandomPointPrompter(Prompter):
    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
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
        return {'type': 'point', 'frame': frame_idx, 'points': random_point}


class MaskPrompter:
    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
    ):
        if frame_idx % self.annotation_every_n == 0:
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=ann_obj_id,
                mask=mask_bool,
            )
            return {"type": "mask", "frame": frame_idx, "mask": mask_bool}


def maximal_inscribed_circle(binary_label):
    # Convert the label image to a binary image
    binary_label = binary_label.astype(np.uint8)

    dist_map = cv2.distanceTransform(binary_label, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    return center, radius
