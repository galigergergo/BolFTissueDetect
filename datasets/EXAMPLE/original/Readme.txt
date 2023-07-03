File from:
[2] Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels. Han, Chu, et al., Medical Image Analysis 80, 2022

We release a weakly-supervised tissue semantic segmentation dataset for lung adenocarcinoma, named LUAD-HistoSeg.
This dataset aims to use only patch-level annotations to achieve pixel-level semantic segmentation for four tissue categories, tumor epithelial (TE), tumor-associated stroma (TAS), necrosis (NEC) and lymphocyte (LYM). The details of this dataset are shown below:

# Folder Structure

original/
    |_train/                                        * Training set with patch-level annotations
    |   |_1031280-2300-27920-[1 0 0 1].png          * Patches cropped from WSIs in 10X magnification. 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
    |   |_......                                    * The total number of patches in the training set is 16,678
    |
    |_val/                                          * Validation set with pixel-level annotations
    |   |_img/ 
    |   |   |_385277-44539-46781.png                * Patches cropped from WSIs in 10X magnification. 'patient_ID'+'_x-axis'+'_y-axis'.png
    |   |   |_......                                * The total number of patches in the validation set is 300
    |   |_mask/                                     * 
    |       |_385277-44539-46781.png                * The mask share the same filename with the original patch
    |       |_......                                * The total number of mask in the validation set is 300
    |
    |_test/                                         * Test set with pixel-level annotations
    |   |_img/
    |   |   |_387709-10004-55944.png                * Patches cropped from WSIs in 10X magnification. 'patient_ID'+'_x-axis'+'_y-axis'.png
    |   |   |_......                                * The total number of patches in the test set is 307
    |   |
    |   |_mask/
    |       |_387709-10004-55944.png                * The mask share the same filename with the original patch.
    |       |_......                                * The total number of masks in the test set is 307
    |
    |_Readme.txt                                    * Readme file of this dataset
    |_Example_for_using_image-level_label.py        * Source code to read the patch-level labels


# Training set


## Naming convensions
- 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png  -> example: 1031280-2300-27920-[1 0 0 1].png

- [a: Tumor epithelial (TE), b: Necrosis (NEC), c: Lymphocyte (LYM), d: Tumor-associated stroma (TAS)]

- Each image (patch) in the training set was cropped from a WSI image at a random anchor point (x, y), with height (224) and width (224), where (x, y) is the top-left corner of the patch.

# Validation and test sets
In validation and test sets, we provide patches with height (224) and width (224) under 10x magnification with the semantic segmentation labels of each type of tissue in P mode with the following palette:

palette = [0]*15
palette[0:3] = [205,51,51]          # Tumor epithelial (TE)
palette[3:6] = [0,255,0]            # Necrosis (NEC)
palette[6:9] = [65,105,225]         # Lymphocyte (LYM)
palette[9:12] = [255,165,0]         # Tumor-associated stroma (TAS)
palette[12:15] = [255, 255, 255]    # White background or exclude
