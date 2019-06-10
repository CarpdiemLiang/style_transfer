Description: 
==========
This is the README for photo-to-ukiyoe cycle-GAN style transfer task. Over half of the codes are adopted from 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix' and then modified. The rest are written by student. 

Requirements:
==========
Install visdom and dominate if willing to display training progress on a webpage by:
    pip install -visdom
    pip install -dominate

Code Structure:
==========
single_test.ipynb:   run this notebook to show the Ukiyoe-style transfer result of 'test_image.jpg'. Make sure the image, latest_ukiyoe_G_A.pkl and './models' are in their original places
train.ipynb:  run this notebook to train a cycle-GAN that can transfer 'datasets/trainA' style to 'datasets/trainB' style. Training options can be found and revised in './options/train_options.py' and './options/base_options.py'
test.ipynb:  run this notebook to test the model in './checkpoints' file. Input the model name in './options/base_options.py'
plot_losses.ipynb:   run this to plot losses given a loss log in './checkpoints'

./options/base_options.py:   stores basic training and testing options.
./options/train_options.py:   stores other training options
./options/test_options.py:   stores other testing options

./models/base_model.py:   base class of all the models
./models/cycle_gan_model.py:   implement cycle-GAN model
./models/networks.py:   define basic network behavior methods
./models/test_model.py:   define some testing settings and run the testing from test.ipynb

./util/:   include python files that handle data loading and processing, webpage display and image buffer.

./datasets/:   a folder that stores training and testing data in trainA, trainB, testA and testB subfolders.

./checkpoints/:   a folder storing saved models, loss logs and training options

latest_ukiyoe_G_A.pkl: the saved generator that can translate images into ukiyoe-style, used in single_test.ipynb

test_image.jpg: test image used in single_test.ipynb


