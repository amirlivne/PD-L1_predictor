DL-based-PDL1-predictor
-----
Pre-trained model to predict PD-L1 status based on IHC scans from H&E images, as described in the paper "Deep Learning Based Image Analysis Predicts PDL1 Status From H&E-Stained Histopathology Images in Breast Cancer".


![Model Architecture](https://github.com/amirlivne/amirlivne.github.io/blob/main/PDL%D6%B91_arch.png?raw=true)


The supplied code infers the PD-L1 status from JPEG images of Hematoxylin and Eosin TMA scans. 
The status is given by a probability score in range `[0,1]`.  

Examples for Hematoxylin and Eosin images are available under `data_examples`.

![Model Architecture](https://github.com/amirlivne/amirlivne.github.io/blob/main/H&E_images.png?raw=true)


Usage
-----

    Usage:
        predict_on_folder.py [arguments] [options]
    
    Arguments:
       <images_root_dir>      The path to the directory containing the .jpg images for inference.

    Options:
        --model_path          The path to the pre-trained model for inference (a '.pt' file). 
                              The default path points to the model that was trained and evaluated as descrived
                              in the published paper.
        --output_file         A file path for saving the outut results. If no file is given, the results will
                              be printed in the terminal, but won't be saved to a file.
Examples
-------
Run with default parameters:


    $ python3 predict_on_folder.py data_examples

Saving results to an output file:

    $ python3 predict_on_folder.py data_examples --output_file results.txt

Predict on a different directory:

    $ python3 predict_on_folder.py <path to a directory> --output_file results.txt

Requirements
------- 
    Python 3.7 and above.
    
    opencv_python==4.5.5.64
    torch==1.10.2+cu113
    torchvision==0.11.3+cu113
    tqdm==4.64.0

To install the requirements, use:
    
    $ pip3 install -r requirements.txt