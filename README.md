# Project Introduction
Our project contains three different models, one is in "cycle_gan_unet" directory which uses the u-net like cnn as generators, one is in "Ukiyoe_codes" directory which uses Resnet blocks as generators, the other is in neural_style_transfer that implement sytle transfer using convolution neural network proposed in this paper https://arxiv.org/pdf/1508.06576.pdf.

## Cycle-Gan-Unet
 Requirements:
=================
Download the check-points for the model from the google drive link, and put them into the corresponding directorys.
/baroque/checkpoint.pth.tar: https://drive.google.com/open?id=1oMTewhni1L7ZW0F9nNgNoE2RfkrGZ500
/ukiyo_e/checkpoint.pth.tar: https://drive.google.com/open?id=1mEQliUwOKgSLSUuB_vBXwl03HH_p4VJO
/salience_model/model.ckpt-200.data-00000-of-00001: https://drive.google.com/open?id=1u8gW2Oj8lZ_Cxqg561lQR9ioDaK64LwX

### Structure:
=========================
/baroque                         -- Store the checkpoints for baroque style translation
/ukiyo_e                             -- Store the checkpoints for ukiyo_e style translation
/meta_grapsh                         -- Store the information of the salient objective detection model
/salience_model                      -- Store the checkpoints for salient objective detection model
demo.ipynb                           -- This notebook is used for demo
cycle_gan_unet                       -- This notebook is the main function of the model
nntools.py                           -- This .py file abstract the status manager and realize the training process of the model
util.py                              -- This .py file is used to realize the image pool called by nntools.py
inference.py                         -- This .py file is used to run the pretrained salient objective detection model
.pkl                                 -- All the pickle files are used to store the images

### Usage:
===============
Directly run the demo.ipynb notebook. You can see the original image and the transferred image.
If you want to train the model by yourself, delete /baroque and /ukiyo_e directorys. And run the cycle_gan_model.ipynb notebook. You can set all the parameters in the initialization of the experiment class.


## Neural Style Transfer: 
### Requirements: 
=========================================================================================
Install package 'pillow' as: $ pip install --user pillow <br/>
Install package 'matplotlib' as: $ pip install --user matplotlib

### Structure:
=========================================================================================
Neural_Style_Transfer.ipynb      -- Run neural style transfer method<br/>
/images                          -- Store the style image and content image for this part

### Usage:
=========================================================================================
Change the network structure by choosing content_layers_default and style_layers_default 
and commenting oghers. For white noise input, consider decreasing the weight of style
loss and increase the number of optimizing steps. 


