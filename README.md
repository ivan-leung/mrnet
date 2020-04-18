# mrnet
Predict knee injuries from MRI image stacks using CNN and transfer learning.

This is the submission code (with updated documentation and installation instructions) to Stanford ML Group's MRNet Dataset competition: https://stanfordmlgroup.github.io/competitions/mrnet/. To download the full dataset of 1,000+ MRI examinations, register on the Stanford ML site using the same link above.

The submission was made in Apr 2019 under IL_baseline (single model) and achieved a 0.900 AUC (the leading model achieved a 0.917 AUC). See the leaderboaord using the link above for the latest ranking.

## MRI Image Data
Each MRI examination consists of 3 image stacks, one each for the sagittal, axial, and coronal planes. Each of the 3 image stacks consists of 20-40 grayscale MRI images of dimension 256x256. 3 types of diagnosis (abnormality, ACL tear, and meniscal tear) are provided.

## Model Training
Transfer learning was used to generate image features to train the model. This simple architecture was selected to leverage the high-level understanding of images that the transfer learning model possesses, and to overcome hardware limitations in training complex models without GPU access. The ResnNt50 pre-trained on ImageNet was used as the transfer learning model (https://keras.io/applications/#resnet). MaxPooling and concatenation operations were performed on the feature vectors (which were obtained by passing the images through the pre-trained model) to produce the input features of the same dimension (3x2048) for each MRI examination.
To train the model quickly (given the hardware constraints), a dense layer immediately follows the input features to model the diagnoses. The trained models for each of the 3 diagnoses were saved under the data/models directory.

## Diagnoses Prediction
The validation and test data are similarly pre-processed using the pre-trained ResNet50 model and maxpooling and concatenation steps described above. Using the saved models, the diagnoses were predicted and logged for submissions. To test the diagnoses using these models:

- Obtain the MRnet Dataset through the Stanford ML group (https://stanfordmlgroup.github.io/competitions/mrnet/) and select the MRI examinations to be diagnosed. Save the files under the `data/MRNet-v1.0/valid`. 
- Save the paths of the .npy files corresponding to the MRI examinations on `src/input_data.csv` using the saggital, coronal, axial ordering.
- Run the code via Docker using the command `sh run.sh`
