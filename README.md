Git repo for `Protecting IoT Devices from Malware with Thermal Side-Channel Anomaly Detection`

## Environment setup

To set up the environement for the code, follow the following instructions.

- Set up the prerequisites
    ```
    pip install -r requirements.txt
    ```
    
## Data Pre-process

- Download the video dataset [here.](https://drive.google.com/file/d/1VpCCVLGJHyeyEJ3_J8ELU27RdIvtD_8Z/view?usp=sharing)

- Extract the video dataset.
    ```
    unzip video_dataset.zip
    ```

- Pre-process the data
    ```
    python data_preprocess.py
    ```
    
## Train models

- To train the AutoEncoder model, please run the following.
    ```
    python autoencoder.py
    ```
- To train the CNN model, please run the following.
    ```
    python simple_cnn.py
    ```    

## Inference

For inference, please run the `Inference.ipynb` notebook by setting the `video_file` to the path of the video being tested, and `model_path` to the path of the saved model.

## Misc

- For an example of the Point Spread Function (PSF) and BRISQUE evaluation, please run the `psf.ipynb` notebook.

## Contact

If you have any questions regarding this repo, please contact vinoj@umd.edu or anubhav@umd.edu
