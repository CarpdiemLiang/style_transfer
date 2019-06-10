# Project Introduction
Our project contains three different models, one is in "cycle_gan_unet" directory which uses the u-net like cnn as generators, one is in "Ukiyoe_codes" directory which uses Resnet blocks as generators, the other is in neural_style_transfer that implement sytle transfer using convolution neural network proposed in this paper https://arxiv.org/pdf/1508.06576.pdf.

## Cycle-Gan-Unet
### Requirements:
Download the check-points for the model from the google drive link, and put them into the corresponding directorys.<br/>
/baroque/checkpoint.pth.tar: https://drive.google.com/open?id=1oMTewhni1L7ZW0F9nNgNoE2RfkrGZ500<br/>
/ukiyo_e/checkpoint.pth.tar: https://drive.google.com/open?id=1mEQliUwOKgSLSUuB_vBXwl03HH_p4VJO<br/>
/salience_model/model.ckpt-200.data-00000-of-00001: https://drive.google.com/open?id=1u8gW2Oj8lZ_Cxqg561lQR9ioDaK64LwX<br/>

### Structure:
/baroque                         -- Store the checkpoints for baroque style translation<br/>
/ukiyo_e                             -- Store the checkpoints for ukiyo_e style translation<br/>
/meta_grapsh                         -- Store the information of the salient objective detection model<br/>
/salience_model                      -- Store the checkpoints for salient objective detection model<br/>
demo.ipynb                           -- This notebook is used for demo<br/>
cycle_gan_unet                       -- This notebook is the main function of the model<br/>
nntools.py                           -- This .py file abstract the status manager and realize the training process of the model<br/>
util.py                              -- This .py file is used to realize the image pool called by nntools.py<br/>
inference.py                         -- This .py file is used to run the pretrained salient objective detection model<br/>
.pkl                                 -- All the pickle files are used to store the images<br/>

### Usage:
Directly run the demo.ipynb notebook. You can see the original image and the transferred image.
If you want to train the model by yourself, delete /baroque and /ukiyo_e directorys. And run the cycle_gan_model.ipynb notebook. You can set all the parameters in the initialization of the experiment class.


## Neural Style Transfer: 
### Requirements: 
Install package 'pillow' as: $ pip install --user pillow <br/>
Install package 'matplotlib' as: $ pip install --user matplotlib

### Structure:
Neural_Style_Transfer.ipynb      -- Run neural style transfer method<br/>
/images                          -- Store the style image and content image for this part

## Usage:
Change the network structure by choosing content_layers_default and style_layers_default 
and commenting oghers. For white noise input, consider decreasing the weight of style
loss and increase the number of optimizing steps. 


