Embedding Watermarks into Deep Neural Networks
====
This code is the implementation of "Embedding Watermarks into Deep Neural Networks". It embeds a digital watermark into deep neural networks in training the host network. This embedding is achieved by a parameter regularizer.

This README will be updated later for more details.

## Usage
```sh
# train the host network while embedding a watermark
python train_wrn.py config/train_random_min.json

# extract the embedded watermark
python utility/wmark_validate.py result/wrn_WTYPE_random_DIM256_SCALE0.01_N1K4B64EPOCH3_TBLK1.weight result/wrn_WTYPE_random_DIM256_SCALE0.01_N1K4B64EPOCH3_TBLK1_layer7_w.npy result/random

# visualize the embedded watermark
python utility/draw_histogram_signature.py config/draw_histogram.json hist_signature.png
```
