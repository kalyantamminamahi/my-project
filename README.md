# Automated Breast Cancer Detection Using Histopathology
Images and Deep Learning Techniques

Aim Of the Project: The primary aim of this project is to develop an automated detection system for
breast cancer that leverages histopathology images and deep learning methods. By doing so, I hope
to achieve a higher level of accuracy and efficiency in the identification of Invasive Ductal Carcinoma,
ultimately improving diagnostic processes in clinical settings.
## Project Structure

The project is organized as follows:

- **UploadedFolder:** Contains the dataset of breast cancer histology images.
- **saved_models:** Stores the trained models in Keras format (.keras).
- **ProjectCode.ipynb:** Jupyter Notebook containing the code for data loading, preprocessing, model building, training, and evaluation.


## Data

The dataset consists of breast cancer histology images labeled as IDC(+) or IDC(-). Images are loaded, resized, and normalized before being used for model training and evaluation. Class imbalance is addressed using RandomUnderSampler.


## Models

The project explores three different CNN models:

1. **Custom CNN:** A simple CNN architecture with convolutional, max-pooling, and dropout layers.
2. **VGG16:** A pre-trained VGG16 model with fine-tuning for IDC classification.
3. **ResNet50:** A pre-trained ResNet50 model with fine-tuning for IDC classification.


## Usage

To run the project:

1. Open the Jupyter Notebook (`ProjectCode.ipynb`).
2. Execute the code cells sequentially to load the data, train the models, and evaluate their performance.
3. Use the `files.upload()` function to upload a new image for prediction. The best model based on the previous evaluation will be used for prediction, and the results will be displayed.


## Results

The notebook provides visualizations of the models' architectures using `visualkeras`, learning curves, and confusion matrices. Classification reports including precision, recall, F1-score, and support are also generated for each model. The best model is selected based on the weighted average F1-score.


## Dependencies

The project requires the following libraries:

- TensorFlow
- Keras
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- imblearn
- visualkeras
- PIL (Pillow)
- IPython

