#
Cell Cell Interaction Gan: Counterfactual Hypothesis Testing of Tumor Microenvironment Scenarios Through Semantic Image Synthesis

## Dependencies, Library installation

pytorch, sklearn, skimage, Pillow, and other required library versions can be found in [pytorch_p36](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-pytorch.html) version in Amazon EC2.

Alternatively, if running on local computer, using Conda, install Python 3.6, pytorch 1.8.1, pillow 8.1.2, opencv 3.4.1, numpy 1.19.2 


## Instructions

Raw data can be downloaded from the TNBC MIBI, Lung Cancer t-CyCIF, and Colorectal Cancer CODEX datasets mentioned in the manuscript.

Then raw multiplexed image data can be processed using the data processing iPython notebooks into square semantic image maps and corresponding "real" multiplexed images.

Training on a 8-12 GB GPU should take a couple hours (5 hours max), depending on the batch size. You can train a CCIGAN model in CCIGAN.ipynb

## Test Data

Test data consisting of pickled numpy arrays of TNBC MIBI 64x64 can be found in test_data folder. These can be loaded into the Dataloader class in the model ipynbs (such as CCIGAN.ipynb). You can test inference on the model by loading in the model weights from "ccigan_mibi_120e" into the Generator model class.
