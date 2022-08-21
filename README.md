DL-based-PDL1-predictor
-----
Pre-trained model to predict PD-L1 status based on IHC scans from H&E images, as described in the paper Deep learning-based image analysis predicts PDL1 status from H&E-stained histopathology images in breast cancer.


![Model Architecture](https://github.com/amirlivne/amirlivne.github.io/blob/main/PDL%D6%B91_arch.png?raw=true)


The supplied code infers the PD-L1 status from JPEG images. 
The status is given by a probability score in range `[0,1]`.  

Examples for Hematoxylin and Eosin images are available under `data_examples`.

![Model Architecture](https://github.com/amirlivne/amirlivne.github.io/blob/main/H&E_images.png?raw=true)


Usage
-----

    Usage:
        predict_on_folder.py [arguments] [options]
    
    Arguments:
       images_root_dir        The path to the directory containing the .jpg images for inference.

    Options:
        --model_path          The path to the pre-trained model for inference (a '.pt' file). 
                              The defult path points to the model that was trained and evaluated as descrived in the paper
                              `Deep learning-based image analysis predicts PDL1 status from H&E-stained histopathology images in breast cancer.`
        --output_file         A file path for saving the outut results.
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