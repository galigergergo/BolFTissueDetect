import Augmentor
import os


__all__ = ['augment_n_times']


def augment_wsi_data(input_folder, output_folder, sample_amount):
    p = Augmentor.Pipeline(input_folder, output_folder)
    p.skew_left_right(probability=1.0, magnitude=0.05)
    p.rotate(probability=1, max_left_rotation=3, max_right_rotation=3)
    p.rotate_random_90(probability=0.75)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.sample(sample_amount)


def augment_n_times(dataset_path, n):
    for root_folder in [f.name for f in os.scandir(dataset_path)
                        if f.is_dir()]:
        root_path = os.path.join(dataset_path, root_folder)
        for class_folder in [f.name for f in os.scandir(root_path)
                             if f.is_dir()]:
            input_folder = os.path.join(root_path, class_folder)
            in_size = len([name for name in os.listdir(input_folder)])
            output_folder = os.path.normpath(os.path.join(dataset_path,
                                                          os.pardir))
            dataset_name = os.path.basename(output_folder)
            output_folder = os.path.normpath(os.path.join(output_folder,
                                                          os.pardir,
                                                          dataset_name +
                                                          '_aug',
                                                          'data', root_folder,
                                                          class_folder))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            augment_wsi_data(os.path.abspath(input_folder),
                             os.path.abspath(output_folder), 10 * in_size)
