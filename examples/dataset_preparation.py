import os
from binbagnets.data import conversions, augmentation


if __name__ == "__main__":
    orig_path = 'datasets/EXAMPLE/original'
    imgnet_path = 'datasets/EXAMPLE/imagenet'
    bin_path = 'datasets/EXAMPLE/binary'

    # initialize dataset type
    conversions.init_dataset('LUAD')

    # convert original format dataset to ImageNet format
    conversions.format_luad_to_imagenet(orig_path, imgnet_path)

    # create binary datasets from ImageNet format dataset
    for part in ['train', 'test', 'val']:
        for clss in range(4):
            print()
            print(conversions.CLASSES[clss], '-', part)
            conversions.create_binary_dataset(imgnet_path, bin_path, clss,
                                              part)
    
    # augment all binary datasets 10x
    for path in [f.name for f in os.scandir(bin_path) if f.is_dir()]:
        augmentation.augment_n_times(os.path.join(bin_path, path, 'data'), 10)
