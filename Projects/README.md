## Here you can the Dataset format and high level instructions to run Holoported Characaters


- Holoported Characaters: Real-time Free-viewpoint Rendering of Humans from Sparse RGB Cameras (Shetty et al. 2024 (CVPR 2024))(Projects/HoloportedCharacters)
  - [`Projects/HoloportedCharacters`](Projects/HoloportedCharacters) 
- General Recommendation
  - Process dataset for 100 frames (as instructed below)
  - Run testing code using our pretrained weights
  - Run training code
Move on to training on entire dataset
## More general tools are also here:

- General utilities with functions that are used in the projects
  - [`Projects/AdditionalUtils`](Projects/AdditionalUtils) 

- Some useful network architectures
  - [`Projects/Architectures`](Projects/Architectures) 
  - In particular it contains: 
    - ResNet50 (He et al. 2015) - [more info](https://arxiv.org/abs/1512.03385)  
    - pix2pix (Isola et al. 2017) - [more info](https://www.tensorflow.org/tutorials/generative/pix2pix)
    - Structure-aware GraphConvs (Habermann et al. 2021) - [more info](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/)
     - ESRNet - [more info](https://arxiv.org/abs/1707.02921)

- Cuda Renderer (implemented in C++/ Cuda with Tensorflow python modules)
  - [`Projects/CudaRenderer`](Projects/CudaRenderer) 
  - This builds the python interface for the C++/Cuda renderer

- Custom tensorflow operators
  - [`Projects/CustomTFOperators`](Projects/CustomTFOperators) 
  - This builds the python interface for the C++/Cuda custom operators


- Generate textures 
  - [`Projects/GenerateTextures`](Projects/GenerateTextures) 
  - Algorithm for texture stitching from multiple views 
## Dataset Summary

Here is a table summarizing the dataset

| Subject ID | Subject Name | Type  | Motion Type  | DOFs | Hands | Number of Cameras | Training Frames | Testing Frames | Tracked Meshes | NEUS2 Reconstructio | Input Views    | Testing Views |
|------------|--------------|-------|--------------|------|-------|-------------------|-----------------|----------------|----------------|---------------------|----------------|---------------|
| S1         | Vlad         | Tight | General      | 54   | No    | 101               | 18000           | 7000           | Yes            | Soon                | [88,58,65,28]  | [40,7,18,27]  |
| S2         | Franzi       | Loose | General      | 62   | No    | 94                | 19000           | 7000           | Yes            | Soon                | [62,25,39,53]  | [40,7,18,27]  |
| S3         | Oleks        | Tight | General      | 107  | Yes   | 116               | 25000           | 9000           | Yes            | Soon                | [104,66,31,77] | [7,22,29,74]  |
| S4         | Heming       | Loose | Free Talking | 107  | Yes   | 116               | 7400            | 3000           | Yes            | Soon                | [102,5,76,31]  | [7,22,29,74]  |
| S5         | Ashwath      | Tight | Free Talking | 107  | Yes   | 116               | 15000           | 1200           | Yes            | Soon                | [103,66,77,31] | [7,22,29,74]  |
## Dataset structure 
The data can be downloaded from https://gvv-assets.mpi-inf.mpg.de/
This data contains zips for the subjects used in the paper 
S1 (Vlad), S2 (Franzi(Loose)), S3 (Oleks), S4 (Heming (free talking)), S5 (Ashwath (free talking))

We also release a folder containing the calibrations for each subject seperately 

Unzipping any one of them would release the data associated with a subject
The structure of a data is as follows


Subject name
- simplified.obj is the embedded graph  
- .character file (pointing to template, skinning weights and rigging pose )
- segmentation.txt used to compute rigidity coefficients
- training/
    
- - skeletoolToGTPose  (3D Pose tracking and normalized motion)
- - videos (Multi View Videos) 
- - mesh_sequence  (Character model output)
- - segmentation_videos (Multi View Segmentation Videos)

## Data Processing (sample instructions for Oleks subject)
### General recommendation, Process 200 frames (0-200), and get everything to work, and then train on all of the data

Step 1: The goal of this step to dump the images, and segmentation masks from each view so that they can be used 

- `Projects/AdditionalUtils/create_frames.py` (takes in camera number as input and dumps the frames into a output folder, edit the paths according to your dataset directory)
- `Projects/AdditionalUtils/create_seg_masks.py` (takes in camera number as input and dumps the segmentation into a output folder, edit the paths according to your dataset directory)

Sample bash script to process them `Projects/AdditionalUtils/submit_bash_data_gen2.sh` 

Note: This process can be done in many ways, the general goal is to dump the frames and the segmentation masks, if you are writing your own code just follow the naming format of ours, it should make life easier for you down the line


Step 2 This step will dump the output of our character model as pointclouds so that it can be used, for training. 
- We release the output of our character model as a .meshes file (Full character model will be released soon )
- To process this file run `Projects/AdditionalUtils/process_mesh_sq_to_pc.py` (set the start and end frame according to the number of frames in a subject) (In this case I recommend to dump all the outputs)

Step 3 This step generates the partial textures according to the four views
- Update `Projects/Settings/SettingsTMap` according to how you processed the data 
- Run `Projects/GenerateTextures/texture_mapping_optimized_general.py` (sample bash script generate_textures_fast.sh) (Note set start and stop to the amount of frames you processed,I have set it to 100-200 in this case)

Step 3 Generate TFRecords, this loads the motions and files into a format compatible w TF 
- Run `python BashCreateTFRecordDataset.py 116 54 100 200 /CT/ashwath2/static00/DatasetRelease/Oleks/training/ 1` (This generates shuffled data for training)
- Update script according to how you processed the data, 
- Run `python BashCreateTFRecordDataset.py 116 54 100 200 /CT/ashwath2/static00/DatasetRelease/Oleks/training/ 0` (This generates unshuffled data for testing)



## Training

cd into `Projects/HoloportedCharacters/`
Create a folder named `SlurmCodeBackUp` 
- Edit `Configs/Oleks/oleks_sr_tex_4k.sh` according to your processed data structure
- Run Bash.py --config Projects HoloportedCharacters/Configs/Oleks/oleks_sr_tex_4k.sh to train (shfiles/Oleks/submitBashAshTexSRFull.sh provides a sample script)

We train the model till convergence, usually (100k) iterations


If the loss is reducing for frames 100-200, I would guess everything is setup right and you can proceed with full training

## Testing
- Edit `Configs/Oleks/oleks_tex_test_4k.sh` according to your processed data structure and where you saved the weights [`or use the weights we provide`](https://drive.google.com/drive/folders/1a_NOXZUmdR5KcphGG9UWb6ms2hqFFsZV?usp=sharing) 
- run testing_scripts/test_data_loader_spiral_more_start_end.py --config testing_scripts/config.ini (for spirals)
- run testing_scripts/test_data_loader_spiral_more_start_end.py --config testing_scripts/fixed_cameras_config.ini (for fixed_camera)