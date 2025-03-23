import random
import cv2
import numpy as np
import numpy.typing as npt
from abc import abstractmethod

random.seed(0)


def maximal_inscribed_circle(binary_label):
    # Convert the label image to a binary image
    binary_label = binary_label.astype(np.uint8)

    dist_map = cv2.distanceTransform(
        binary_label, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    return center, radius


def add_point_prompts(
    predictor, inference_state, ann_obj_id, frame_index, points, labels
):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_index,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )


def filter_points_outside_mask(
    points_in_yx: npt.NDArray[np.int64], mask_bool: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    if not points_in_yx.size:
        return points_in_yx
    return points_in_yx[mask_bool[points_in_yx[:, 0], points_in_yx[:, 1]]]


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


class KConsistentPointPrompter(Prompter):
    def __init__(self, annotation_every_n: int, k=10):
        super().__init__(annotation_every_n)
        self.prev_center_xy = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : 1x2
        self.prev_positive_prompts_yx = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : kx2
        self.k = k

    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
    ):
        """Add a consistent point prompt.

        This method saves `prev_point` which is the point prompted in the previous frame (i.e. frame_idx - 1).

        Pick a new point prompt every first frame of annotation.
        """
        should_add_point_prompt = frame_idx % self.annotation_every_n != 0
        if not should_add_point_prompt:
            return None

        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        mask_coords = np.argwhere(mask_bool)
        self.prev_center_xy = filter_points_outside_mask(
            self.prev_center_xy[:, ::-1], mask_bool
        )[:, ::-1]
        self.prev_positive_prompts_yx = filter_points_outside_mask(
            self.prev_positive_prompts_yx, mask_bool
        )

        no_positive_points_in_mask = len(mask_coords) == 0
        if no_positive_points_in_mask:
            # Reset the center to empty 2d array
            self.prev_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            # Reset the other positive prompts to empty 2d array
            self.prev_positive_prompts_yx = np.ndarray(
                shape=(0, 2), dtype=np.int32
            )
            return None

        is_first_frame_for_point_prompt = (
            frame_idx % self.annotation_every_n == 1
        )
        if is_first_frame_for_point_prompt or not self.prev_center_xy.size:
            center, _ = maximal_inscribed_circle(mask_bool)
            self.prev_center_xy = np.array([center])
        else:  # if not the first frame, we should add points prompts
            n_sample_points = min(self.k, len(mask_coords))
            n_positive_prompts = self.prev_positive_prompts_yx.shape[0]
            positive_prompts_yx = np.array(
                random.choices(
                    mask_coords, k=n_sample_points - n_positive_prompts
                ),
                dtype=np.int32,
            )
            positive_prompts_yx = positive_prompts_yx.reshape(-1, 2)
            self.prev_positive_prompts_yx = np.concatenate(
                [self.prev_positive_prompts_yx, positive_prompts_yx], axis=0
            )

        points = np.concatenate(
            [
                self.prev_center_xy,
                self.prev_positive_prompts_yx[
                    :, ::-1
                ],  # convert from (y, x) to (x, y)
            ],
            axis=0,
        )
        labels = np.ones(points.shape[0], dtype=np.int32)
        add_point_prompts(
            predictor, inference_state, ann_obj_id, frame_idx, points, labels
        )

        self.prev_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
        return {"type": "point", "frame": frame_idx, "points": points}


class ConsistentPointPrompter(KConsistentPointPrompter):
    """A prompter that put the same point from the previous frame."""

    def __init__(
        self,
        annotation_every_n: int,
    ):
        # Set k = 0, so no additional point except the max inscribed circle's center is prompted
        super().__init__(self, annotation_every_n=annotation_every_n, k=0)


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
        return {"type": "point", "frame": frame_idx, "points": random_point}


class MaskPrompter(Prompter):
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
