#
Cell Cell Interaction Gan: Counterfactual Hypothesis Testing of Tumor Microenvironment Scenarios Through Semantic Image Synthesis

## Dependencies

pytorch, sklearn, skimage, Pillow, and other required library versions can be found in pytorch_p36 version in Amazon EC2


## Instructions

Raw data can be downloaded from the TNBC MIBI, Lung Cancer t-CyCIF, and Colorectal Cancer CODEX datasets mentioned in the manuscript.

Then raw multiplexed image data can be processed using the data processing iPython notebooks into square semantic image maps and corresponding "real" multiplexed images.

Training on a 8-12 GB GPU should take a couple hours (5 hours max), depending on the batch size. You can train a CCIGAN model in CCIGAN.ipynb
