import os
import shutil
import random
from PIL import Image
import numpy as np

__all__ = ['format_luad_to_imagenet', 'create_binary_dataset', 'init_dataset',
           'CLASSES']

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


def get_labels_from_file_name(file_path):
    """
    Get class labels of a file from the file name of the following format:
        {file_name}-[0 1 0 0].png,
        where the last part of the file name corresponds to the class labels

    Parameters
    ----------
    file_path : string
        Absolute path to file.

    Returns
    -------
    file_name : string
        The name of the file without the class labels.
    labels : list
        A list of the class labels containing 0 and 1 values.
    """
    file = os.path.basename(file_path)[:-4]
    file_name = file[:-10]
    label_str = file.split(']')[0].split('[')[-1].split(' ')
    labels = [int(label_str[0]), int(label_str[1]),
              int(label_str[2]), int(label_str[3])]
    return file_name, labels


def get_labels_from_file_name_tight(file_path):
    """
    Get class labels of a file from the file name of the following format:
        {file_name}[0100].png,
        where the last part of the file name corresponds to the class labels

    Parameters
    ----------
    file_path : string
        Absolute path to file.

    Returns
    -------
    file_name : string
        The name of the file without the class labels.
    labels : list
        A list of the class labels containing 0 and 1 values.
    """
    file = os.path.basename(file_path)[:-4]
    file_name = file[:-6]
    label_str = file.split(']')[0].split('[')[-1]
    labels = [int(label_str[0]), int(label_str[1]),
              int(label_str[2]), int(label_str[3])]
    return file_name, labels


def get_labels_from_mask_file(file_path):
    """
    Get class labels of a file from the corresponding mask file, which contians
    different pixel colors for classes.

    Parameters
    ----------
    file_path : string
        Absolute path to file.

    Returns
    -------
    file_name : string
        The name of the file without the class labels.
    labels : list
        A list of the class labels containing 0 and 1 values.
    """
    file_name = os.path.basename(file_path)
    mask_file_name = os.path.normpath(os.path.join(file_path, os.pardir,
                                                   os.pardir, 'mask',
                                                   file_name))
    img = np.array(Image.open(mask_file_name))
    labels = np.setdiff1d(np.unique(img), NON_CLASS_MASK_LABELS)
    labels = [CLASS_MASK_LABELS.index(lbl) for lbl in labels]
    labels = [1 if i in labels else 0 for i in range(len(CLASSES))]
    return file_name[:-4], labels


def get_labels_from_imagenet_format(root_dir, file_name):
    """
    Get class labels of a file from an IMAGENet format dataset from the
    name of all the directories containing it.

    Parameters
    ----------
    dataset_path : string
        Absolute path to root directory of dataset.
    file_name : string
        Name of the file.

    Returns
    -------
    file_name : string
        The name of the file without the class labels.
    labels : list
        A list of the class labels containing 0 and 1 values.
    """
    labels = [1 if os.path.exists(os.path.join(root_dir, clss, file_name))
              else 0 for clss in CLASSES]
    return file_name[:-4], labels


def format_luad_to_imagenet(orig_path, target_path):
    """
    Creates a new version of the LUAD dataset in ImageNet format:
        - root
            - class 1
                - img 1
                - img 2
                ...
            - class 2
                - img 1
                - img 2
                ...
            ...

    Parameters
    ----------
    orig_path : string
        Path to root directory of LUAD-HistoSeg dataset.
    target_path : string
        Path to desired root directory of formatted dataset.
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for root_folder in [f.name for f in os.scandir(orig_path) if f.is_dir()]:
        if not os.path.exists(os.path.join(target_path, root_folder)):
            os.makedirs(os.path.join(target_path, root_folder))
        for classs in CLASSES:
            if not os.path.exists(os.path.join(target_path, root_folder,
                                               classs)):
                os.makedirs(os.path.join(target_path, root_folder, classs))

        file_names = [fn for fn
                      in os.listdir(os.path.join(orig_path, root_folder))
                      if fn.endswith('png')]
        root_folder_ext = root_folder
        if not len(file_names):
            root_folder_ext = os.path.join(root_folder, 'img')
            file_names = [fn for fn in 
                          os.listdir(os.path.join(orig_path, root_folder_ext))
                          if fn.endswith('png')]

        for file in file_names:
            file_path = os.path.join(orig_path, root_folder_ext, file)
            if os.path.basename(root_folder_ext) == 'img':
                file_name, labels = get_labels_from_mask_file(file_path)
            else:
                # LUAD-histoseg
                file_name, labels = get_labels_from_file_name(file_path)
                # BCSS-WSSS
                # file_name, labels = \
                #    get_labels_from_file_name_tight(file_path)

            for i in range(len(CLASSES)):
                if labels[i]:
                    shutil.copyfile(os.path.join(orig_path, root_folder_ext,
                                                 file),
                                    os.path.join(target_path, root_folder,
                                                 CLASSES[i],
                                                 file_name + '.png'))


def get_file_names_of_class_from_file_names(dataset_path, clss):
    """
    Get a list of all the file names that contain a certain class and a list of
    all the file names that do not contain that same class from a dataset
    folder with classes specified in the file names as in the LUAD-HistoSeg
    dataset.

    Parameters
    ----------
    dataset_path : string
        Path to root directory of dataset (train subdirectory).
    clss : int
        Index of class in the file name.
    
    """
    print('\tGetting separating file names... ', end='')
    containing_lst = []
    not_containing_lst = []
    file_names = [fn for fn in os.listdir(dataset_path) if fn.endswith('png')]
    for file in file_names:
        file_name = file[:-4]
        label_str = file_name.split(']')[0].split('[')[-1].split(' ')
        label = int(label_str[clss])
        if label:
            containing_lst.append(file)
        else:
            not_containing_lst.append(file)
    print('DONE')
    return containing_lst, not_containing_lst


def get_file_names_of_class_from_imagenet(dataset_path, clss):
    """
    Get a list of all the file names that contain a certain class and a list of
    all the file names that do not contain that same class from a dataset
    folder of ImageNet format.


    Parameters
    ----------
    dataset_path : string
        Path to a subdirectory (train, test, val) of dataset.
    clss : int
        Index of class in the file name.
    
    """
    print('\tGetting separating file names... ', end='')
    containing_lst = []
    not_containing_lst = []
    
    file_names = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_names.append(file)

    for file in np.unique(file_names):
        file_name, labels = get_labels_from_imagenet_format(dataset_path, file)
        if labels[clss]:
            cls_file = os.path.join(CLASSES[clss], file)
            containing_lst.append(cls_file)
        else:
            cls_file = os.path.join(CLASSES[labels.index(1)], file)
            not_containing_lst.append(cls_file)
    
    print('DONE')
    return containing_lst, not_containing_lst


def create_binary_dataset(dataset_path, target_path,
                          clss, part_dir='train'):
    """
    Creates a binary classification dataset for a single class from a dataset
    folder with classes specified in the file names as in the LUAD-HistoSeg
    dataset.

    Parameters
    ----------
    dataset_path : string
        Path to root directory of dataset.
    target_path : string
        Path to desired root directory of the newly created dataset.
    clss : int
        Index of class in the file name.
    part_dir : string
        Subdirectory of dataset (train, test, val).
    
    """
    dataset_part_path = os.path.join(dataset_path, part_dir)

    print('Creating binary dataset...')
    containing_lst, not_containing_lst = \
        get_file_names_of_class_from_imagenet(dataset_part_path, clss)
    
    # sample from larger list roughly the same amount as from the smaller list
    # resulting dataset size will be round (eg. divisible by 1000) 
    print('\tSampling larger subdataset... ', end='')
    small_len = min(len(containing_lst), len(not_containing_lst))
    nr_digits = len(str(small_len)) - 1
    sampled_len = 10**nr_digits * (small_len //
                                   10**nr_digits + 1) * 2 - small_len
    large_lst, cont_samp = (not_containing_lst, 0) if \
        len(containing_lst) == small_len else (containing_lst, 1)
    while len(large_lst) < sampled_len:
        sampled_len -= 10**nr_digits
    sampled_lst = random.sample(large_lst, sampled_len)
    print('DONE')

    # save dataset in ImageNet format of positive (containing) and
    # negative (not containing) classes.
    print('\tCopying files to binary dataset... ', end='')
    target_path = os.path.join(target_path, CLASSES[clss], 'data', part_dir)
    pos_path = os.path.join(target_path, 'pos')
    if not os.path.exists(pos_path):
        os.makedirs(pos_path)
    pos_lst = sampled_lst if cont_samp else containing_lst
    for file in pos_lst:
        shutil.copyfile(os.path.join(dataset_part_path, file),
                        os.path.join(pos_path, os.path.basename(file)))
    neg_path = os.path.join(target_path, 'neg')
    if not os.path.exists(neg_path):
        os.makedirs(neg_path)
    neg_lst = not_containing_lst if cont_samp else sampled_lst
    for file in neg_lst:
        shutil.copyfile(os.path.join(dataset_part_path, file),
                        os.path.join(neg_path, os.path.basename(file)))
    print('DONE')
    print('ALL DONE')
