from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
from binbagnets.models import pytorchnet
from binbagnets.models.utils import generate_heatmap_pytorch


__all__ = ['init_dataset', 'plot_heatmap_analysis']

# LUAD-histoseg
CLASSES_LUAD = ['TE', 'NEC', 'LYM', 'TAS']
CLASS_MASK_LABELS_LUAD = [0, 1, 2, 3]
NON_CLASS_MASK_LABELS_LUAD = [4]

# BCSS-WSSS
CLASSES_BCSS = ['TUM', 'STR', 'LYM', 'NEC']
CLASS_MASK_LABELS_BCSS = [0, 1, 2, 3]
NON_CLASS_MASK_LABELS_BCSS = [4]

DATASET = None
CLASSES = None
CLASS_MASK_LABELS = None
NON_CLASS_MASK_LABELS = None


def init_dataset(dataset_type):
    """
    Initialize global dataset variables for the later functions.

    Parameters
    ----------
    dataset_type : string
        Type of the dataset, from ['LUAD', 'BCSS'].
    """
    global DATASET, CLASSES, CLASS_MASK_LABELS, NON_CLASS_MASK_LABELS
    if dataset_type == 'LUAD':
        DATASET = 'LUAD'
        CLASSES = CLASSES_LUAD
        CLASS_MASK_LABELS = CLASS_MASK_LABELS_LUAD
        NON_CLASS_MASK_LABELS = NON_CLASS_MASK_LABELS_LUAD
    elif dataset_type == 'BCSS':
        DATASET = 'BCSS'
        CLASSES = CLASSES_BCSS
        CLASS_MASK_LABELS = CLASS_MASK_LABELS_BCSS
        NON_CLASS_MASK_LABELS = NON_CLASS_MASK_LABELS_BCSS


def load_pretrained_binbagnet(model_name, model_path, prefix):
    """
    Load a pretrained binary BagNet model from checkpoint file.
    Remove prefixes if model state dict is wrapped in some
    object (e.g. module.).

    Parameters
    ----------
    model_name : string
        Name of binary BagNet model, from ['binbagnet9', 'binbagnet17',
                                           'binbagnet33', 'binbagnet_small'].
    model_path : string
        Absolute path to checkpoint file.
    prefix : string
        Prefix of the state dict wrapper object (e.g. module.).

    Returns
    -------
    model : torch model
        Loaded pretrained torch model.
    """
    # load file
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    loaded_dict = checkpoint['state_dict']
    print('Loading pretrained BagNet model from...')
    print('\tpath: %s' % model_path)
    print('\tepoch: %d' % checkpoint['epoch'])
    
    # strip state dict from prefixes if present
    n_clip = len(prefix)
    adapted_dict = {(k[n_clip:] if k.startswith(prefix) else k): v
                    for k, v in loaded_dict.items()}

    # load state dict into BagNet17 model
    model = pytorchnet.__dict__[model_name]()
    model.load_state_dict(adapted_dict)

    return model


def load_image_for_bagnet17(img_path):
    """
    Load image from a file and reshape it as input for a pretrained BagNet17
    model.

    Parameters
    ----------
    img_path : string
        Absolute path to image file.

    Returns
    -------
    image : 4D numpy array
        Loaded and reshaped image.
    """
    print('Loading pretrained image from...')
    print('\tpath: %s' % img_path)

    # load image
    image_orig = imread(img_path)

    # reshape image
    image = np.transpose(image_orig, (2, 0, 1))
    image = np.expand_dims(image, 0)

    return image


def get_binary_mask_for_class(file_path, clss):
    """
    Get binary mask for a segmented file for a specific class.

    Parameters
    ----------
    file_path : string
        Absolute path to file.
    clss : int
        Index of the class in CLASS_MASK_LABELS list.

    Returns
    -------
    mask : 2D numpy array
        Binary mask for the specified image and class.
    """
    file_name = os.path.basename(file_path)
    mask_file_name = os.path.normpath(os.path.join(file_path, os.pardir,
                                                   os.pardir, 'mask',
                                                   file_name))
    mask = np.array(Image.open(mask_file_name))
    
    labels = CLASS_MASK_LABELS.copy()
    labels.remove(labels[clss])

    mask[mask == CLASS_MASK_LABELS[clss]] = 999
    for c in labels:
        mask[mask == c] = 0
    for c in NON_CLASS_MASK_LABELS:
        mask[mask == c] = 0
    mask[mask == 999] = 1
    
    return mask


def calculate_patch_pixel_no_on_img(patch_coords, patch_size, img_size):
    """
    Calculate the number of pixels of a patch overlapping an image,
    excluding pixels that lie outside of image borders.

    Parameters
    ----------
    patch_coords : (int, int) tuple
        Coordinates of center point of patch.
    patch_size : int
        Size of quare shaped patch (length of sides).
    img_size : (int, int) tuple
        Size of the image (width, height).
    
    Returns
    -------
    patch_pixel_no : int
        Number of pixels of a patch on an image.
    """
    patch_radius = (patch_size - 1) // 2

    out_of_edge_x = 0
    if patch_coords[0] < patch_radius:
        out_of_edge_x = patch_radius - patch_coords[0]
    elif img_size[0] - patch_coords[0] < patch_radius:
        out_of_edge_x = patch_radius - img_size[0] + patch_coords[0]

    out_of_edge_y = 0
    if patch_coords[1] < patch_radius:
        out_of_edge_y = patch_radius - patch_coords[1]
    elif img_size[1] - patch_coords[1] < patch_radius:
        out_of_edge_y = patch_radius - img_size[1] + patch_coords[1]

    return (patch_size - out_of_edge_x) * (patch_size - out_of_edge_y)


def calculate_patch_in_mask_percentage(patch_coords, patch_size, bin_mask):
    """
    Calculate what percentage of a given patch lies inside a binary mask.
    If there are parts of the patch outside the picture, do not consider them.

    Parameters
    ----------
    patch_coords : (int, int) tuple
        Coordinates of center point of patch.
    patch_size : int
        Size of quare shaped patch (length of sides).
    mask : 2D numpy array
        Binary mask for a specified image and class.
    
    Returns
    -------
    in_perc : float in range [0, 1]
        Percentage value of the patch inside the given binary mask.
    """
    img_size = bin_mask.shape
    patch_radius = (patch_size - 1) // 2
    in_mask_no = 0
    in_img_no = 0
    for i in range(patch_coords[0] - patch_radius, patch_coords[0] +
                   patch_radius + 1):
        if i >= 0 and i < img_size[0]:
            for j in range(patch_coords[1] - patch_radius, patch_coords[1] +
                           patch_radius + 1):
                if j >= 0 and j < img_size[1]:
                    in_img_no += 1
                    if bin_mask[i][j]:
                        in_mask_no += 1 
    return in_mask_no / in_img_no


def get_most_important_patches(heatmap, most_imp_perc_bound):
    """
    Extract most important patches from a heatmap as a certain percentage of
    all the patches. 

    Parameters
    ----------
    heatmap : 2D numpy array
        Generated heatmap for the specified image.
    most_imp_perc_bound : float in range [0, 1]
        Percentage of most important patches from all patches.
    
    Returns
    -------
    most_imp_patches : list of (int, int) tuples
        A list of coordinates of the most important patches (x, y).
    """
    heatmap_local = np.copy(heatmap)
    img_size = heatmap_local.shape
    all_patch_no = img_size[0] * img_size[1]
    most_imp_patch_no = int(np.floor(all_patch_no * most_imp_perc_bound))
    most_imp_patches = []
    for i in range(most_imp_patch_no):
        x, y = np.unravel_index(heatmap_local.argmax(), img_size)
        most_imp_patches.append((x, y))
        heatmap_local[x][y] = -100000
    return most_imp_patches


def calculate_most_important_patches_in_mask_perc(img_path, clss,
                                                  heatmap, patch_size,
                                                  most_imp_perc_bound,
                                                  ptch_in_mask_perc_bnd):
    """
    Calculate what percentage of the most important patches lies inside a
    binary mask.
    Also parametrized:
        - how many patches are considered most important patches from all
          patches (percentage)
        - what percentage of a patch inside binary mask is considered inside
          mask

    Parameters
    ----------
    img_path : string
        Absolute path to img file.
    clss : int
        Index of the class in CLASS_MASK_LABELS list.
    heatmap : 2D numpy array
        Generated heatmap for the specified image.
    patch_size : int
        Size of quare shaped patch (length of sides).
    most_imp_perc_bound : float in range [0, 1]
        Percentage of most important patches from all patches.
    patch_in_mask_perc_bound : float in range [0, 1]
        Percentage of patch considered inside mask.
    
    Returns
    -------
    in_perc : float in range [0, 1]
        Percentage value of the patch inside the given binary mask.
    inside_patches : 2D numpy array
        A binary array containing most important patches inside binary mask.
    """
    bin_mask = get_binary_mask_for_class(img_path, clss)
    most_imp_patches = get_most_important_patches(heatmap, most_imp_perc_bound)
    patch_in_mask_no = 0
    inside_patches = np.zeros(bin_mask.shape)
    for _, (x, y) in enumerate(most_imp_patches):
        perc = calculate_patch_in_mask_percentage((x, y), patch_size, bin_mask)
        if perc >= ptch_in_mask_perc_bnd:
            inside_patches[x][y] = 1
            patch_in_mask_no += 1
        else:
            inside_patches[x][y] = 2
 
    return patch_in_mask_no / len(most_imp_patches), inside_patches


def plot_heatmap_analysis(img_path, model_name, model_path, clss):
    """
    Creates the heatmap analysis plot for a WSI patch and a specific tissue
    type (class).

    Parameters
    ----------
    img_path : string
        Path to img file.
    model_name : string
        Name of binary BagNet model, from ['binbagnet9', 'binbagnet17',
                                           'binbagnet33', 'binbagnet_small'].
    model_path : string
        Path to file containing trained model state dict.
    clss : int
        Index of the class to generate the image for.
    
    """
    prefix = 'module.'
    bin_mask = get_binary_mask_for_class(img_path, clss)
    image_orig = np.array(Image.open(img_path))

    image_bagnet = load_image_for_bagnet17(img_path)
    model = load_pretrained_binbagnet(model_name, model_path, prefix)
    
    print('Generating heatmap for the selected patch...')
    heatmap = generate_heatmap_pytorch(model, image_bagnet, 0, 17)

    print('Finding most important patches...')
    most_imp_perc_bound = 0.05
    inside_perc, inside_patches = \
        calculate_most_important_patches_in_mask_perc(img_path, clss,
                                                      heatmap, 17,
                                                      most_imp_perc_bound,
                                                      0.1)

    print('Creating heatmap analysis plot...')
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 7))
    for row in axs:
        for a in row:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_xticks([])
            a.set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    fig.suptitle('%.2f%% of the top %.1f%% of patches are in the binary mask' %
                 (inside_perc * 100, most_imp_perc_bound * 100),
                 fontsize=12, y=0.93)

    axs[0][0].imshow(image_orig, interpolation='none')

    axs[0][1].imshow(bin_mask, interpolation='none')

    axs[1][0].imshow(inside_patches, interpolation='none')

    abs_max = np.percentile(np.abs(heatmap), 90)
    axs[1][1].imshow(heatmap, interpolation='none', cmap='RdBu_r',
                     vmin=-abs_max, vmax=abs_max)
    axs[1][1].imshow(bin_mask, interpolation='none', cmap='Oranges', alpha=0.2)

    plt.show()
