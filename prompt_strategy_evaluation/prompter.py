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
        neg_mask_bool: npt.NDArray,
    ):
        pass


class KConsistentPointPrompter(Prompter):
    def __init__(self, annotation_every_n: int, k=10):
        super().__init__(annotation_every_n)
        self.prev_center_xy = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : 1x2
        self.prev_prompts_yx = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : kx2
        self.k = k

    def process_prev_points(self, mask_bool):
        """Process the previous points to keep those inside the mask.

        This allows point prompts to be consistent and not be outside the mask.

        Args:
            neg_mask_bool (npt.NDArray): The negative mask boolean array.
        """
        self.prev_center_xy = filter_points_outside_mask(
            self.prev_center_xy[:, ::-1], mask_bool
        )[:, ::-1]
        self.prev_prompts_yx = filter_points_outside_mask(
            self.prev_prompts_yx, mask_bool
        )

    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
        neg_mask_bool: npt.NDArray,
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

        self.process_prev_points(mask_bool)

        no_positive_points_in_mask = len(mask_coords) == 0
        if no_positive_points_in_mask:
            # Reset the center to empty 2d array
            self.prev_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            # Reset the other positive prompts to empty 2d array
            self.prev_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            return None

        is_first_frame_for_point_prompt = (
            frame_idx % self.annotation_every_n == 1
        )
        if is_first_frame_for_point_prompt or not self.prev_center_xy.size:
            center, _ = maximal_inscribed_circle(mask_bool)
            self.prev_center_xy = np.array([center])

        # TODO: should we even check for the first frame? we always need to add points prompts
        if (
            not is_first_frame_for_point_prompt
        ):  # if not the first frame, we should add points prompts
            n_sample_points = min(self.k, len(mask_coords))
            n_positive_prompts = self.prev_prompts_yx.shape[0]
            positive_prompts_yx = np.array(
                random.choices(
                    mask_coords, k=n_sample_points - n_positive_prompts
                ),
                dtype=np.int32,
            )
            positive_prompts_yx = positive_prompts_yx.reshape(-1, 2)
            self.prev_prompts_yx = np.concatenate(
                [self.prev_prompts_yx, positive_prompts_yx], axis=0
            )

        points = np.concatenate(
            [
                self.prev_center_xy,
                self.prev_prompts_yx[:, ::-1],  # convert from (y, x) to (x, y)
            ],
            axis=0,
        )
        labels = np.ones(points.shape[0], dtype=np.int32)
        add_point_prompts(
            predictor, inference_state, ann_obj_id, frame_idx, points, labels
        )

        self.prev_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
        return {"type": "point", "frame": frame_idx, "points": points}


class RandomPointPrompter(Prompter):
    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
        neg_mask_bool: npt.NDArray,
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


class KNegativeConsistentPointsPrompter(KConsistentPointPrompter):
    def __init__(self, annotation_every_n: int, k=10):
        super().__init__(annotation_every_n, k)
        self.prev_center_xy = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : 1x2
        self.prev_prompts_yx = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : kx2
        self.k = k

    def is_no_negative_points_in_mask(self, mask_coords):
        return len(mask_coords) == 0

    def should_add_point_prompt(self, frame_idx):
        return frame_idx % self.annotation_every_n != 0

    def should_skip(self, mask_coords, frame_idx):
        return self.is_no_negative_points_in_mask(
            mask_coords
        ) or not self.should_add_point_prompt(frame_idx)

    def reset_prev_points(self):
        # Reset the center to empty 2d array
        self.prev_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
        # Reset the other positive prompts to empty 2d array
        self.prev_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)

    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
        neg_mask_bool: npt.NDArray,
    ):
        """Add negative consistent point prompts.

        This method saves `prev_point` which is the point prompted in the previous frame (i.e. frame_idx - 1).

        Pick a new point prompt every first frame of annotation.
        """
        mask_coords = np.argwhere(neg_mask_bool)
        if self.should_skip(mask_coords, frame_idx):
            return None

        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.process_prev_points(neg_mask_bool)

        no_neg_points_in_mask = len(mask_coords) == 0
        if no_neg_points_in_mask:
            self.reset_prev_points()
            return None

        is_first_frame_for_point_prompt = (
            frame_idx % self.annotation_every_n == 1
        )
        if is_first_frame_for_point_prompt or not self.prev_center_xy.size:
            center, _ = maximal_inscribed_circle(neg_mask_bool)
            self.prev_center_xy = np.array([center])

        # TODO: should we even check for the first frame? we always need to add points prompts
        if (
            not is_first_frame_for_point_prompt
        ):  # if not the first frame, we should add points prompts
            n_sample_points = min(self.k, len(mask_coords))
            n_positive_prompts = self.prev_prompts_yx.shape[0]
            positive_prompts_yx = np.array(
                random.choices(
                    mask_coords, k=n_sample_points - n_positive_prompts
                ),
                dtype=np.int32,
            )
            positive_prompts_yx = positive_prompts_yx.reshape(-1, 2)
            self.prev_prompts_yx = np.concatenate(
                [self.prev_prompts_yx, positive_prompts_yx], axis=0
            )

        points = np.concatenate(
            [
                self.prev_center_xy,
                self.prev_prompts_yx[:, ::-1],  # convert from (y, x) to (x, y)
            ],
            axis=0,
        )
        labels = np.zeros(points.shape[0], dtype=np.int32)
        add_point_prompts(
            predictor, inference_state, ann_obj_id, frame_idx, points, labels
        )

        self.prev_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
        return {"type": "point", "frame": frame_idx, "neg_points": points}


class KBorderPointsPrompter(Prompter):
    """A prompter that prompts negative points on the border of the mask."""

    def __init__(self, annotation_every_n: int, pos_k=10, neg_k=10):
        super().__init__(annotation_every_n)

        # Consistent positive points
        self.prev_pos_center_xy = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : 1x2
        self.prev_pos_prompts_yx = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : (pos_k-1)x2
        self.pos_k = pos_k

        # Consistent negative points
        self.prev_neg_center_xy = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : 1x2
        self.prev_neg_prompts_yx = np.ndarray(
            shape=(0, 2), dtype=np.int32
        )  # when not empty : (neg_k-1)x2
        self.neg_k = neg_k

    def process_prev_points(self, mask_bool, prev_center_xy, prev_prompts_yx):
        """Process the previous points to keep those inside the mask.

        This allows point prompts to be consistent and not be outside the mask.

        Args:
            neg_mask_bool (npt.NDArray): The negative mask boolean array.
        """
        if prev_center_xy.size:
            prev_center_xy = filter_points_outside_mask(
                prev_center_xy[:, ::-1], mask_bool
            )[:, ::-1]
        if prev_prompts_yx.size:
            prev_prompts_yx = filter_points_outside_mask(
                prev_prompts_yx, mask_bool
            )
        return prev_center_xy, prev_prompts_yx

    def process_negative_prompts(
        self,
        border_coords: npt.NDArray,
        neg_mask_bool: npt.NDArray,
    ):
        """Get negative prompts on the border of the mask."""
        # Keep only the points that are inside the mask
        self.prev_neg_center_xy, self.prev_neg_prompts_yx = (
            self.process_prev_points(
                neg_mask_bool, self.prev_neg_center_xy, self.prev_neg_prompts_yx
            )
        )

        no_neg_points_in_mask = neg_mask_bool.sum() == 0
        if no_neg_points_in_mask:
            # Reset the center to empty 2d array
            self.prev_neg_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            # Reset the other positive prompts to empty 2d array
            self.prev_neg_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            return None

        # Center
        if not self.prev_neg_center_xy.size:
            center, _ = maximal_inscribed_circle(neg_mask_bool)
            self.prev_neg_center_xy = np.array([center])

        # Prompts
        n_sample_points = min(self.neg_k, len(border_coords))
        n_neg_prompts = self.prev_neg_prompts_yx.shape[0]
        neg_prompts_yx = np.array(
            random.choices(border_coords, k=n_sample_points - n_neg_prompts),
            dtype=np.int32,
        )

        neg_prompts_yx = neg_prompts_yx.reshape(-1, 2)
        self.prev_neg_prompts_yx = np.concatenate(
            [self.prev_neg_prompts_yx, neg_prompts_yx], axis=0
        )

    def process_positive_prompts(
        self,
        border_coords: npt.NDArray,
        pos_mask_bool: npt.NDArray,
    ):
        """Get positive prompts on the border of the mask."""
        # Keep only the points that are inside the mask
        self.prev_pos_center_xy, self.prev_pos_prompts_yx = (
            self.process_prev_points(
                pos_mask_bool, self.prev_pos_center_xy, self.prev_pos_prompts_yx
            )
        )

        no_pos_points_in_mask = pos_mask_bool.sum() == 0
        if no_pos_points_in_mask:
            # Reset the center to empty 2d array
            self.prev_pos_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            # Reset the other positive prompts to empty 2d array
            self.prev_pos_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            return None

        # Center
        if not self.prev_pos_center_xy.size:
            center, _ = maximal_inscribed_circle(pos_mask_bool)
            self.prev_pos_center_xy = np.array([center])

        # Prompts
        n_sample_points = min(self.pos_k, len(border_coords))
        n_pos_prompts = self.prev_pos_prompts_yx.shape[0]
        positive_prompts_yx = np.array(
            random.choices(border_coords, k=n_sample_points - n_pos_prompts),
            dtype=np.int32,
        )
        positive_prompts_yx = positive_prompts_yx.reshape(-1, 2)
        self.prev_pos_prompts_yx = np.concatenate(
            [self.prev_pos_prompts_yx, positive_prompts_yx], axis=0
        )

    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
        neg_mask_bool: npt.NDArray,
    ):
        """Add negative consistent point prompts.

        This method saves `prev_point` which is the point prompted in the previous frame (i.e. frame_idx - 1).

        Pick a new point prompt every first frame of annotation.
        """

        # Make sure the border is not too close to the edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        neg_mask_bool = cv2.erode(
            neg_mask_bool.astype(np.uint8), kernel, iterations=3
        ).astype(np.bool)

        # Get the border of the negative mask next to the positive mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(
            mask_bool.astype(np.uint8), kernel, iterations=10
        )  # Enlarge the positive mask
        border = cv2.bitwise_and(
            dilated_mask, neg_mask_bool.astype(np.uint8)
        )  # Get the border of the negative mask next to the positive mask
        border_coords = np.argwhere(border)
        if len(border_coords) != 0:
            self.process_negative_prompts(
                border_coords=border_coords, neg_mask_bool=neg_mask_bool
            )

        # Get the border of the positive mask
        # pos_border = cv2.Canny(mask_bool.astype(np.uint8), 0, 1)
        # pos_border_coords = np.argwhere(pos_border)
        pos_coords = np.argwhere(mask_bool)
        if len(pos_coords) != 0:
            self.process_positive_prompts(
                border_coords=pos_coords, pos_mask_bool=mask_bool
            )

        # Reset the prev pos and neg points if there are no points in the mask
        mask_coords = np.argwhere(mask_bool)
        no_points_in_mask = len(mask_coords) == 0
        if no_points_in_mask:
            self.prev_pos_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_pos_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_neg_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_neg_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            return None

        # Prompt
        final_prompt = {
            "type": "point",
            "frame": frame_idx,
        }
        if (
            not self.prev_neg_center_xy.size
            and not self.prev_neg_prompts_yx.size
            and not self.prev_pos_center_xy.size
            and not self.prev_pos_prompts_yx.size
        ):
            return None

        # Add the negative prompts
        if self.prev_neg_center_xy.size or self.prev_neg_prompts_yx.size:
            neg_points = np.concatenate(
                [
                    self.prev_neg_center_xy,
                    self.prev_neg_prompts_yx[
                        :, ::-1
                    ],  # convert from (y, x) to (x, y)
                ],
                axis=0,
            )
            neg_labels = np.zeros(neg_points.shape[0], dtype=np.int32)
            add_point_prompts(
                predictor=predictor,
                inference_state=inference_state,
                ann_obj_id=ann_obj_id,
                frame_index=frame_idx,
                points=neg_points,
                labels=neg_labels,
            )
            final_prompt["neg_points"] = neg_points

        if self.prev_pos_center_xy.size or self.prev_pos_prompts_yx.size:
            # Add the positive prompts
            pos_points = np.concatenate(
                [
                    self.prev_pos_center_xy,
                    self.prev_pos_prompts_yx[
                        :, ::-1
                    ],  # convert from (y, x) to (x, y)
                ],
                axis=0,
            )
            pos_labels = np.ones(pos_points.shape[0], dtype=np.int32)
            add_point_prompts(
                predictor=predictor,
                inference_state=inference_state,
                ann_obj_id=ann_obj_id,
                frame_index=frame_idx,
                points=pos_points,
                labels=pos_labels,
            )
            final_prompt["points"] = pos_points
        return final_prompt


class KBorderPointsPrompterV2(KBorderPointsPrompter):
    def __init__(self, annotation_every_n: int, pos_k=10, neg_k=10):
        """Enfore positive and negative points to be next to each other."""
        super().__init__(annotation_every_n, pos_k, neg_k)

    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
        neg_mask_bool: npt.NDArray,
    ):
        """Add negative consistent point prompts.

        This method saves `prev_point` which is the point prompted in the previous frame (i.e. frame_idx - 1).

        Pick a new point prompt every first frame of annotation.
        """

        # Make sure the border is not too close to the edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(
            mask_bool.astype(np.uint8), kernel, iterations=15
        ).astype(
            np.bool
        )  # Enlarge the positive mask
        dilated_dilated_mask = cv2.dilate(
            dilated_mask.astype(np.uint8), kernel, iterations=15
        ).astype(
            np.bool
        )  # Enlarge the positive mask

        neg_border = dilated_dilated_mask & ~dilated_mask
        neg_border = (
            neg_border & neg_mask_bool
        )  # Ensure that dilated masks are in the negative mask

        neg_border_coords = np.argwhere(neg_border)
        if len(neg_border_coords) != 0:
            self.process_negative_prompts(
                border_coords=neg_border_coords, neg_mask_bool=neg_mask_bool
            )

        # Get the border of the positive mask
        eroded_mask = cv2.erode(
            mask_bool.astype(np.uint8), kernel, iterations=15
        ).astype(np.bool)
        pos_border = mask_bool & ~eroded_mask

        pos_border_coords = np.argwhere(pos_border)
        if len(pos_border_coords) != 0:
            self.process_positive_prompts(
                border_coords=pos_border_coords, pos_mask_bool=mask_bool
            )

        # Reset the prev pos and neg points if there are no points in the mask
        mask_coords = np.argwhere(mask_bool)
        no_points_in_mask = len(mask_coords) == 0
        if no_points_in_mask:
            self.prev_pos_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_pos_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_neg_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_neg_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            return None

        # Prompt
        final_prompt = {
            "type": "point",
            "frame": frame_idx,
        }
        if (
            not self.prev_neg_center_xy.size
            and not self.prev_neg_prompts_yx.size
            and not self.prev_pos_center_xy.size
            and not self.prev_pos_prompts_yx.size
        ):
            return None

        # Add the negative prompts
        if self.prev_neg_center_xy.size or self.prev_neg_prompts_yx.size:
            neg_points = np.concatenate(
                [
                    self.prev_neg_center_xy,
                    self.prev_neg_prompts_yx[
                        :, ::-1
                    ],  # convert from (y, x) to (x, y)
                ],
                axis=0,
            )
            neg_labels = np.zeros(neg_points.shape[0], dtype=np.int32)
            add_point_prompts(
                predictor=predictor,
                inference_state=inference_state,
                ann_obj_id=ann_obj_id,
                frame_index=frame_idx,
                points=neg_points,
                labels=neg_labels,
            )
            final_prompt["neg_points"] = neg_points

        if self.prev_pos_center_xy.size or self.prev_pos_prompts_yx.size:
            # Add the positive prompts
            pos_points = np.concatenate(
                [
                    self.prev_pos_center_xy,
                    self.prev_pos_prompts_yx[
                        :, ::-1
                    ],  # convert from (y, x) to (x, y)
                ],
                axis=0,
            )
            pos_labels = np.ones(pos_points.shape[0], dtype=np.int32)
            add_point_prompts(
                predictor=predictor,
                inference_state=inference_state,
                ann_obj_id=ann_obj_id,
                frame_index=frame_idx,
                points=pos_points,
                labels=pos_labels,
            )
            final_prompt["points"] = pos_points
        return final_prompt


class KBorderPointsPrompterV3(KBorderPointsPrompter):
    def __init__(self, annotation_every_n: int, pos_k=10, neg_k=10):
        """Enfore positive and negative points to be next to each other."""
        super().__init__(annotation_every_n, pos_k, neg_k)

    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
        neg_mask_bool: npt.NDArray,
    ):
        """Add negative consistent point prompts.

        This method saves `prev_point` which is the point prompted in the previous frame (i.e. frame_idx - 1).

        Pick a new point prompt every first frame of annotation.
        """

        # Make sure the border is not too close to the edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(
            mask_bool.astype(np.uint8), kernel, iterations=15
        ).astype(
            np.bool
        )  # Enlarge the positive mask
        dilated_dilated_mask = cv2.dilate(
            dilated_mask.astype(np.uint8), kernel, iterations=15
        ).astype(
            np.bool
        )  # Enlarge the positive mask

        neg_border = dilated_dilated_mask & ~dilated_mask

        neg_border_coords = np.argwhere(neg_border)
        if len(neg_border_coords) != 0:
            self.process_negative_prompts(
                border_coords=neg_border_coords, neg_mask_bool=neg_border
            )

        # Get the border of the positive mask
        eroded_mask = cv2.erode(
            mask_bool.astype(np.uint8), kernel, iterations=15
        ).astype(np.bool)
        pos_border = mask_bool & ~eroded_mask

        pos_border_coords = np.argwhere(pos_border)
        if len(pos_border_coords) != 0:
            self.process_positive_prompts(
                border_coords=pos_border_coords, pos_mask_bool=pos_border
            )

        # Reset the prev pos and neg points if there are no points in the mask
        mask_coords = np.argwhere(mask_bool)
        no_points_in_mask = len(mask_coords) == 0
        if no_points_in_mask:
            self.prev_pos_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_pos_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_neg_center_xy = np.ndarray(shape=(0, 2), dtype=np.int32)
            self.prev_neg_prompts_yx = np.ndarray(shape=(0, 2), dtype=np.int32)
            return None

        # Prompt
        final_prompt = {
            "type": "point",
            "frame": frame_idx,
        }
        if (
            not self.prev_neg_center_xy.size
            and not self.prev_neg_prompts_yx.size
            and not self.prev_pos_center_xy.size
            and not self.prev_pos_prompts_yx.size
        ):
            return None

        # Add the negative prompts
        if self.prev_neg_center_xy.size or self.prev_neg_prompts_yx.size:
            neg_points = np.concatenate(
                [
                    self.prev_neg_center_xy,
                    self.prev_neg_prompts_yx[
                        :, ::-1
                    ],  # convert from (y, x) to (x, y)
                ],
                axis=0,
            )
            if neg_points.shape[0] < self.neg_k:
                print("neg_points", neg_points, flush=True)
                raise ValueError(
                    "neg_points.shape[0]",
                    neg_points.shape[0],
                    "self.prev_neg_center_xy",
                    self.prev_neg_center_xy,
                    "self.prev_pos_center_xy",
                    self.prev_pos_center_xy,
                )
            neg_labels = np.zeros(neg_points.shape[0], dtype=np.int32)
            add_point_prompts(
                predictor=predictor,
                inference_state=inference_state,
                ann_obj_id=ann_obj_id,
                frame_index=frame_idx,
                points=neg_points,
                labels=neg_labels,
            )
            final_prompt["neg_points"] = neg_points

        if self.prev_pos_center_xy.size or self.prev_pos_prompts_yx.size:
            # Add the positive prompts
            pos_points = np.concatenate(
                [
                    self.prev_pos_center_xy,
                    self.prev_pos_prompts_yx[
                        :, ::-1
                    ],  # convert from (y, x) to (x, y)
                ],
                axis=0,
            )
            pos_labels = np.ones(pos_points.shape[0], dtype=np.int32)
            add_point_prompts(
                predictor=predictor,
                inference_state=inference_state,
                ann_obj_id=ann_obj_id,
                frame_index=frame_idx,
                points=pos_points,
                labels=pos_labels,
            )
            final_prompt["points"] = pos_points
        return final_prompt


class MaskPrompter(Prompter):
    def add_prompt(
        self,
        predictor,
        inference_state,
        frame_idx: int,
        ann_obj_id: int,
        mask_bool: npt.NDArray,
        neg_mask_bool: npt.NDArray,
    ):
        if frame_idx % self.annotation_every_n == 0:
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=ann_obj_id,
                mask=mask_bool,
            )
            return {"type": "mask", "frame": frame_idx, "mask": mask_bool}
