# Advanced Convolutions

### Acheive 85% accuracy on CIFAR10 dataset, using Cutout augmentantion and dilated and depthwise convloutions in the model(<200,000 paramaeters and no Max  Pooling)

**Aim**

- ✅ change the code such that it uses GPU 

- ✅change the architecture to C1C2C3C40  (No MaxPooling, but 3 3x3 layers with stride of 2 instead)  
- ✅total RF must be more than 44 
- ✅one of the layers must use Depthwise Separable Convolution 
- ✅one of the layers must use Dilated Convolution 
- ✅use GAP (compulsory):- add FC after GAP to target #of classes (optional) 
- use albumentation library and apply:
  - ✅horizontal flip 
  - ✅shiftScaleRotate 
  - ✅coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

- ✅achieve 85% accuracy, as many epochs as you want.
- ✅Total Params to be less than 200k. 


# File Structure
 1. utils.py - Contains helper function to get the correct mean+std deviation statistics of the dataset
2. models.py - CNN model architecture with less than 200k parameters with a capability to get 88.5 % accuracy consistently in less than 15 epochs
3. dataloader.py : Contains code for the train and test data loaders for the CIFAR10. Various augmentation are added here as well.
4. train.py : training function
5. test.py : test function
6. Advanced_Convolutions.ipynb : Colab notebook for training on GPU's which imports necessaay classes/functions from above files. all results can be seen here
7. ../Images/ - contains a grid  of sample images after augmentation (Cutout has been succesfully used)

# Joint modules
Dilated and Depth wise convolutions are used whenever we want to reduce the FLOP's or the model size. We might want to reduce the model size because GPU and storage devices equipped on the embedded and mobile terminals cannot support large models. An efficient way to use dilated and depthwise convolutions has been explained in the paper [Lightweight image classifier using dilated and depthwise separable convolutions](https://www.researchgate.net/publication/345401718_Lightweight_image_classifier_using_dilated_and_depthwise_separable_convolutions). They introduce the concept of joint modules where depthwise convolutions follow dilated convolutions as shown in the below image.
![Imgur](https://imgur.com/mpvnSP9.png)

**These joint modules have been used in the last layer of my model as well and have helped me reduce 64960 parameters**: 
(128x64x3x3 - (64x3x3 + 128x64))


# Results
- Model size - 176,234 parmeters (Net4)
- Highest test accuracy - 85.2%
- First reached 85% in the 42nd epoch 

# Group members
Nishant Bhansali

Ruchika Agrawal
 
