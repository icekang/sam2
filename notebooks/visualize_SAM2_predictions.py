from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    _ = ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    _ = ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    return _


annotation_every_n = 4
prompting_method = 'consistent_10+1_point_prompts'
for var in range(1, 5):
    output_name = "sam2.1_hiera_s_MOSE_finetune_scale4_var{}.yaml".format(var)
    for patient_id in ['101-045', '706-005']:
        prediction_output = Path(f'./{output_name}_annotate_every_{annotation_every_n}_{prompting_method}/')
        prediction_dir = prediction_output
        prediction_path = prediction_dir / f'video_segments_{patient_id}.pt'
        predictions = torch.load(prediction_path, weights_only=False)
        
        prompt_path = prediction_dir / f'prompts_{patient_id}.pt'
        prompts = torch.load(prompt_path, weights_only=False)
        frame_idx2prompt = {prompt['frame']: prompt for prompt in prompts}

        groundtruth_dir = Path('/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/labelsTs')
        groundtruth_path = groundtruth_dir / patient_id

        image_dir = Path('/home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/imagesTs')
        image_path = image_dir / patient_id

        color_pallete = np.array([[0, 0, 0], [0, 255, 0], [0, 255, 0]])

        fig, ax = plt.subplots(1, 3, figsize=(6, 2))

        image_paths = list(image_path.glob('*.jpg'))
        image_paths.sort()

        groundtruth_paths = list(groundtruth_path.glob('*.png'))
        groundtruth_paths.sort()

        frame = 0


        ax[1].set_title('Groundtruth')

        def update(frame):
            ax[2].clear()
            ax[0].set_title(f'Image ({frame} | {image_paths[frame].name})')
            
            if frame % annotation_every_n == 0:
                ax[2].set_title('SAM2 Prediction (mask prompted)')
            else:
                ax[2].set_title('SAM2 Prediction')

            image = Image.open(image_paths[frame])
            image = image.convert('RGBA')

            gt = Image.open(groundtruth_paths[frame])
            gt = gt.convert('RGBA')

            prediction = Image.fromarray(color_pallete[predictions[frame][1][0].astype(np.uint8)].astype(np.uint8))
            prediction = prediction.convert('RGBA')

            new_image = Image.blend(gt, prediction, 0.5)
            new_image = Image.blend(image, new_image, 0.5)
            
            ax_2_output = ax[2].imshow(prediction)
            if frame in frame_idx2prompt:
                prompt = frame_idx2prompt[frame]
                prompt_type = prompt['type']
                if prompt_type == 'mask':
                    pass
                elif prompt_type == 'point':
                    points_xy = prompt['points']
                    N, two = points_xy.shape
                    ax_2_output = show_points(
                        coords=points_xy,
                        labels=np.array([1] * N),
                        ax=ax[2], marker_size=200)
            return ax[0].imshow(new_image), ax[1].imshow(gt), ax_2_output

        update(0)
        print(f'\n{patient_id}_annotation_every_n_{annotation_every_n}_{output_name}_{prompting_method}')
        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(image_paths))        
        ani.save(f'{patient_id}_annotation_every_n_{annotation_every_n}_{output_name}_{prompting_method}.mp4', progress_callback=lambda i, n: print('\r', i, '/', n, end=''))
