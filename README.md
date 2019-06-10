# Directory Introduction
Our project contains three different models, one is in "cycle_gan_unet" directory which uses the u-net like cnn as generators, one is in "Ukiyoe_codes" directory which uses Resnet blocks as generators, the other is in ...

## Cycle-Gan-Unet
 Requirements:
=================
Download the check-points for the model from the google drive link, and put them into the corresponding directorys.
/baroque/checkpoint.pth.tar: https://drive.google.com/open?id=1oMTewhni1L7ZW0F9nNgNoE2RfkrGZ500
/ukiyo_e/checkpoint.pth.tar: https://drive.google.com/open?id=1mEQliUwOKgSLSUuB_vBXwl03HH_p4VJO
/salience_model/model.ckpt-200.data-00000-of-00001: https://drive.google.com/open?id=1u8gW2Oj8lZ_Cxqg561lQR9ioDaK64LwX


# Neural Style Transfer: 
  Requirements: 
=========================================================================================
Install package 'pillow' as: $ pip install --user pillow <br/>
Install package 'matplotlib' as: $ pip install --user matplotlib

Structure:
=========================================================================================
Neural_Style_Transfer.ipynb      -- Run neural style transfer method<br/>
/images                          -- Store the style image and content image for this part

Usage:
=========================================================================================
Change the network structure by choosing content_layers_default and style_layers_default 
and commenting oghers. For white noise input, consider decreasing the weight of style
loss and increase the number of optimizing steps. 


