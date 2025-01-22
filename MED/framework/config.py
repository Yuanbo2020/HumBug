import os
from framework.utilities import create_folder

Data_space = r'E:\Yuanbo\Code\26_MED_MSC\0_Dataset'
audio_space = os.path.join(Data_space, 'audio')
metadata_space = os.path.join(Data_space, 'metadata')

feat_type = 'FeatB'

cuda = 1
endswith = '.pth'

# Librosa settings
rate = 8000
win_size = 100
step_size = 50
mel_bins = 64
min_duration = 100  # 100 frames are 1s

# 之前是 30对应1.92s， step=5,对应0.32s
# 现在是 100对应1s, step=50,对应0.5秒

# Calculating window size based on desired min duration (sample chunks)
# default at 8000Hz: 2048 NFFT -> NFFT/4 for window size = hop length in librosa.
# Recommend lowering NFFT to 1024 so that the default hop length is 256 (or 32 ms).
# Then a win size of 60 produces 60x32 = 1.92 (s) chunks for training

########################################### config pytorch #########################################################
dropout = 0.2

# Settings for binary classification for main.ipynb

# ResNet learning rate:
# lr = 0.0015 # Learning rate may need adjusting for multiclass VGGish.

# VGGish learning rate:
lr = 0.0003

max_overrun = 10
epochs = 50
batch_size = 32  # Increased batch size for VGGish (DEBUG)
pretrained = True

# VGG-ish
vggish_model_urls = {'vggish': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth'}

# Settings for multi-class classification with 8 species for species_classification.ipynb
n_classes = 8
epochs = 100





