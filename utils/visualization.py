import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import sys; sys.path.append('../')

DATASET_COLORS = [[0, 0, 0], [223, 87, 188], [160, 221, 255],
                  [130, 106, 237], [200, 121, 255], [255, 183, 255],
                  [0, 144, 193], [113, 137, 255], [230, 232, 230]]
DATASET_COLORS_BINARY = [[0,0,0], [223,87,188]]

def visualize_seg_mask(data_sample: list, main_title: Optional[str] = None, binary=True):
    """
    Drawing semantic segmentation mask and visualize it with original image and masked image.
    
    Parameters
    ----------
    data_sample : list
        list of images (np.ndarray) and their semantic segmentation masks (np.ndarray)
    """
    num_samples = len(data_sample)
    
    fig, axes_list = plt.subplots(nrows=num_samples, ncols=3, figsize=(10, 5))
    plt.subplots_adjust(hspace = 0, wspace = 0)
    
    orig_img, ann_mask, masked_img = axes_list[0][0], axes_list[0][1], axes_list[0][2]
    
    orig_img.set_title('Image', fontsize=10)
    ann_mask.set_title('Annotated mask', fontsize=10)
    masked_img.set_title('Masked image', fontsize=10)

    palette = DATASET_COLORS if not binary else DATASET_COLORS_BINARY
    
    for idx in range(num_samples):
        image, mask = data_sample[idx]
            
        color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[mask == label, :] = color

        masked_image = np.array(image) * 0.5 + color_seg * 0.5 
        masked_image = masked_image.astype(np.uint8)
        
        axes_list[idx][0].imshow(image)
        axes_list[idx][1].imshow(mask)
        axes_list[idx][2].imshow(masked_image)
        
        axes_list[idx][0].set_axis_off()
        axes_list[idx][1].set_axis_off()
        axes_list[idx][2].set_axis_off()
    if main_title:
        plt.suptitle(main_title,
                     x=0.05, y=1.0,
                     horizontalalignment='left',
                     fontweight='semibold',
                     fontsize='large')
    plt.show()