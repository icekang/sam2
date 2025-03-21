import json
import os
from pathlib import Path

import edt
import numpy as np
import SimpleITK as sitk

DATASET_ID = "307"
DATASET_NAME = "Sohee_Calcium_OCT_CrossValidation"
EVERY_N = 16
nnUNet_preprocessed = Path(os.environ["nnUNet_preprocessed"])
nnUNet_raw = Path(os.environ["nnUNet_raw"])

split_filepath = (
    nnUNet_preprocessed
    / f"Dataset{DATASET_ID}_{DATASET_NAME}/splits_final.json"
)
imageTr_dir = nnUNet_raw / f"Dataset{DATASET_ID}_{DATASET_NAME}/imagesTr"
labelTr_dir = nnUNet_raw / f"Dataset{DATASET_ID}_{DATASET_NAME}/labelsTr"

with open(split_filepath, "r") as f:
    splits = json.load(f)

len(splits)


def get_image_path(image_id, is_image=True):
    if is_image:
        return imageTr_dir / f"{image_id}_0000.nii.gz"
    else:
        return labelTr_dir / f"{image_id}.nii.gz"


def read_image(image_filename):
    itk_image = sitk.ReadImage(image_filename)
    npy_image = sitk.GetArrayFromImage(itk_image)
    if npy_image.ndim == 3:
        # 3d, as in original nnunet
        npy_image = npy_image[None]
    elif npy_image.ndim == 4:
        # 4d, multiple modalities in one file
        pass
    else:
        raise RuntimeError(
            f"Unexpected number of dimensions: {npy_image.ndim} in file {image_filename}"
        )

    return npy_image.astype(np.float32)


def iterate_interpolated_edts(image1, image2, n_interpolated):
    # Get the EDT of the forground
    edt1 = edt.sdf(image1)
    edt2 = edt.sdf(image2)

    # Interpolate the EDT
    alphas = np.linspace(0, 1, n_interpolated)
    yield edt1
    for a in alphas:
        edt_i = edt1 * (1 - a) + edt2 * a  # Linear interpolation of the EDT
        yield edt_i
    yield edt2


def interpolate_labels(image1, image2, n_interpolated):
    H, W = image1.shape

    concated_labels = np.concatenate([image1, image2])
    labels = np.unique(concated_labels)
    labels = labels.astype(np.int32)
    edt_for_labels = iterate_interpolated_edts(image1, image2, n_interpolated)
    results = []
    for ith in range(n_interpolated):
        for edt in edt_for_labels:
            results.append(edt > 0)
    return np.stack(results)


def dice_score_of_a_volume(gt, pred, ignore_index=0):
    labels = np.unique(gt)
    dices = []
    for label in labels:
        if label == ignore_index:
            continue
        gt_bool = label == gt
        pred_bool = label == pred

        intersection = np.logical_and(gt_bool, pred_bool)
        dices.append(
            2.0 * intersection.sum() / (gt_bool.sum() + pred_bool.sum())
        )

    avg_dice = sum(dices) / len(dices)
    return labels, dices, avg_dice


pred_dir = Path("sdf_predictions")
pred_dir.mkdir(exist_ok=True)
results = {}
for fold, split in enumerate(splits):
    results[fold] = {}
    # train_ids = split["train"]
    val_ids = split["val"]
    pred_dir = Path(f"sdf_predictions/fold_{fold}")
    pred_dir.mkdir(exist_ok=True)
    for patient_id in val_ids:
        gt_image = read_image(get_image_path(patient_id, is_image=False))
        B, C, H, W = gt_image.shape
        image = gt_image[0, 0::EVERY_N, :, :]

        prediction = np.zeros((B, C, H, W), dtype=np.uint8)
        for i in range(image.shape[0] - 1):
            for frame, label in enumerate(
                interpolate_labels(image[i], image[i + 1], 4)
            ):
                if frame == 0 or frame == 4:
                    label = label.astype(np.int8) * 2
                prediction[0, i * EVERY_N + frame, :, :] = label.astype(np.int8)

        dice = dice_score_of_a_volume(
            gt_image[0, :, :, :], prediction[0, :, :, :]
        )
        results[fold][patient_id] = dice
        prediction = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0]])[prediction]
        # Save the prediction with numpy
        np.savez(pred_dir / f"{patient_id}.npz", prediction)
        print(f"Fold {fold}, Patient {patient_id}, Dice: {dice}")

    #     break
    # break
