# kaggle-carvana-challenge
Project contains solution for [Kaggle Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge).

### Project structure

 - `avg_mask` - baseline solution which applies average masks computed on train images to all test images
 - `lib` - contains common functions / classes / variables used in different solutions
 - `merge_submissions` - solution, which creates new submision merging several existing ones with specified weights
 - `unet_with_cropping` - main solution which uses U-Net neural network for image segmentation
 - `visualization` - contains IPython notebook to visualize predicted masks
