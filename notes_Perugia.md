
### Requirements:
**data:** <br>
separately sent by data transfer (size ~2.4GB). Inquire to rurizzo@med.umich.edu

**data load and processing:** <br>
- Matlab (tested on version R2019a) <br>
- jmrui software (free download for academic institution @ http://www.jmrui.eu/license-and-download/download-2/) <br>
- download code or clone repository @ https://github.com/bellarude/MRSdeeplearning

**deep learning denoising:** <br>
- Python 3.8 runnable environment (suggested Anaconda) <br>
- IDE for python (suggested IntelliJ) <br>
- installation of main libraries for computation, plotting, and deep learning (Matplotlib, scipy, tensorflow) and GPU parallelization strongly recommended <br>
- download code or clone repository @ https://github.com/bellarude/MRSdeeplearning

### Demo step-by-step:
**data load and processing:** <br>
 1. Data conversion for dedicated processing and visualization <br>
	 1.2 Convert philips data -> .mrui (*read_Perugia.m*) <br>
	 1.3 jmrui processing: water removal (HLSVD) + frequency alignment + phase alignment <br>
	 1.4 mrui extra manual phase alignment for each single spectrum <br>
 2.  preparation of data for deep learning algorithm (*read_Perugia_mrui.m*):  <br>
	 2.1 conversion jmrui -> matlab  <br>
	 2.2 saving of .mat tabulated data to be imported in python env. <br>

**deep learning denoising:** <br>
run script *test_unet_denoising.py* <br>
 1. import data 	 <br>
 2. import the model with pre-trained weights (see paper  [Dziadosz M. et al., MRM 90(5), 2023](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29762)) 	 <br>
 3. run the model to produce prediction on newly prov ided data <br>
 4. plot subset of results for visual inspection 	 <br>
 5. saving of input, prediction and errors <br>

**NOTES** <br>
1. all files needs update on *folderPath* and *directory* names according to local settings <br>
2. where and how models are defined can be visualised for denoising purpose @ *model_unet_denoising.py* <br>
3. example for fine-tuning in *ensembling.py* <br>
4. https://www.tensorflow.org/tensorboard for visualization <br>
