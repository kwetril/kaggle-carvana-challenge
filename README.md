# kaggle-carvana-challenge
Project contains solution for [Kaggle Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge).

### Project structure

 - `avg_mask` - baseline solution which applies average masks computed on train images to all test images
 - `lib` - contains common functions / classes / variables used in different solutions
 - `merge_submissions` - solution, which creates new submision merging several existing ones with specified weights
 - `unet_with_cropping` - main solution which uses U-Net neural network for image segmentation
 - `visualization` - contains IPython notebook to visualize predicted masks

Model weights (full images): [carvana_unet.pth](https://yadi.sk/d/LPFECcmF3LtUa9)  
Model weights (crops): [carvana_unet_crop.pth](https://yadi.sk/d/7tudaI0Q3P8cfj)
