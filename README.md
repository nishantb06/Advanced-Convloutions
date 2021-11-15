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

# Results
- Model size - 176,234 parmeters (Net4)
- Highest test accuracy - 85.2%
- First reached 85% in the 42nd epoch 
- Final receptive field - 
 