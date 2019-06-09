# Neural Style Transfer: 
Requirements: 
=========================================================================================
Install package 'pillow' as: $ pip install --user pillow <br/>
Install package 'matplotlib' as: $ pip install --user matplotlib

Structure:
=========================================================================================
Neural_Style_Transfer.ipynb      -- Run neural style transfer method
/images                          -- Store the style image and content image for this part

Usage:
=========================================================================================
Change the network structure by choosing content_layers_default and style_layers_default 
and commenting oghers. For white noise input, considering decreasing the weight of style
loss and increase the number of optimizing steps. 
