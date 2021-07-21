#
Cell Cell Interaction Gan: Counterfactual Hypothesis Testing of Tumor Microenvironment Scenarios Through Semantic Image Synthesis 

[preprint](https://www.biorxiv.org/content/10.1101/2020.10.27.358101v2.abstract)

## Dependencies, Library installation

pytorch, sklearn, skimage, Pillow, and other required library versions can be found in [pytorch_p36](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-pytorch.html) version in Amazon EC2.

Alternatively, if running on local computer, using Conda, install Python 3.6, pytorch 1.8.1, pillow 8.1.2, opencv 3.4.1, numpy 1.19.2 


## Instructions

Raw data can be downloaded from the TNBC MIBI, Lung Cancer t-CyCIF, and Colorectal Cancer CODEX datasets mentioned in the manuscript.

Then raw multiplexed image data can be processed using the data processing iPython notebooks into square semantic image maps and corresponding "real" multiplexed images.

Training on a 8-12 GB GPU should take a couple hours (5 hours max), depending on the batch size. You can train a CCIGAN model in CCIGAN.ipynb


## Demo and Test Data

Test data consisting of pickled numpy arrays of TNBC MIBI 64x64 can be found in test_data folder. These can be loaded into the Dataloader class in the model ipynbs (such as CCIGAN.ipynb). You can test inference on the model by loading in the model weights from "ccigan_mibi_120e" into the Generator model class.

If running on CPU (not GPU) with at least 8GB of RAM, you can load in the model weights provided in the test_data folder

    state = torch.load('../test_data/ccigan_mibi_120e')
    netG.load_state_dict(state['G'])

Then using generator (netG) perform inference on the model by loading the test images into the test dataloader and iterating through the test images

    for idx, data in enumerate(test_set_loader):
        X_seg, X_real = data
        X_seg = torch.clamp(X_seg.transpose(2,1), 0, 1).float().cuda()
        X_real = X_real.transpose(2,1).float().cuda()
        noise = 0.5 * torch.randn(X_seg.size()[0], 128).cuda()
        
        fake = netG(noise, X_seg.detach())
        
        fig=plt.figure(figsize=(2.5, 2.5))
        print("Segmentation: ")
        plt.imshow(seg_show(X_seg.detach().cpu().numpy()[0]))
        plt.show()

        fig=plt.figure(figsize=(16, 10))
        columns = 7
        rows = 4
        print("Fake: ")
        for i in range(24):
            fig.add_subplot(rows, columns, i+1)
            plt.title(channel_names[i])
            plt.imshow(fake.detach().cpu().numpy()[0][i],cmap='hot', interpolation='nearest')
        plt.show()
        
        
        fig=plt.figure(figsize=(16, 10))
        columns = 7
        rows = 4
        print("Real: ")
        for i in range(24):
            fig.add_subplot(rows, columns, i+1)
            plt.title(channel_names[i])
            plt.imshow( X_real[0,i,:,:].detach().cpu().numpy(),cmap='hot', interpolation='nearest')
        plt.show()

To train the model, you would need a GPU otherwise the training time would take more than half a day to a couple days (depending on which patient you trained on).
