# PixelProbe: Image Forgery Detection System 🔍

Welcome to **PixelProbe**! This project is designed to unveil the truth hidden in images by detecting whether an image is authentic or tampered. Using advanced machine learning techniques combined with Error Level Analysis (ELA), PixelProbe reveals alterations in images that might not be visible to the naked eye. Built on a deep learning model, the system is wrapped in a user-friendly interface powered by Streamlit, allowing users to easily upload images and get real-time predictions.

## Features 🌟
- **Upload Image**: Users can upload any image (JPEG, PNG) and have it analyzed in real time.
- **Image Forgery Detection**: The model will classify the uploaded image as either `Authentic` or `Tampered`.
- **Error Level Analysis (ELA)**: Detect image modifications based on compression differences, highlighting parts of the image that have been altered.
- **User-Friendly Interface**: Simple and clean interface powered by Streamlit for easy interaction.
- **Home, About, and Contact Pages**: Navigate easily between different pages to learn more about the system or get in touch with us.

## Demo 💻
You can try out PixelProbe by uploading any image, and within seconds, you’ll receive a detailed prediction on whether the image is `Authentic` or `Tampered`.

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## How It Works 🛠️
PixelProbe uses **Error Level Analysis (ELA)** to highlight regions in an image that have undergone compression, which may indicate tampering. The steps are as follows:
1. **Image Upload**: Upload an image via the provided interface.
2. **ELA Processing**: The image is processed using ELA, which highlights parts of the image with different compression levels.
3. **Model Prediction**: A pre-trained Convolutional Neural Network (CNN) analyzes the processed image and classifies it as `Authentic` or `Tampered`.

## Model Architecture 🧠
The core model is a **Convolutional Neural Network (CNN)** built using Keras and TensorFlow. It includes several layers like `Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Dropout`, and `Dense` layers to ensure robust image classification. The model was trained on the **CASIA v2** dataset, a popular benchmark for image tampering detection.

### Key Components:
- **Error Level Analysis (ELA)** for image preprocessing.
- **Convolutional Neural Network (CNN)** for tampering classification.
- **Streamlit** for frontend and interaction.

## Installation ⚙️

To get started with PixelProbe, follow these steps:


1. **Install Required Libraries**:
   Install the necessary dependencies via `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   After the installation, you can run the Streamlit app using the following command:
   ```bash
   streamlit run app.py
   ```

## Requirements
- Python 3.8+
- TensorFlow 2.10.0+
- Keras 2.10.0
- Numpy 1.23.5
- Matplotlib 3.7.1
- OpenCV 4.8.0
- Pillow 9.5.0
- SciPy 1.10.1
- Scikit-learn 1.2.2
- Streamlit

## Usage 🎮
1. **Home**: Upload an image in JPEG/PNG format to analyze its authenticity.
2. **About**: Learn more about PixelProbe and the technology behind it.
3. **Contact**: Use the form to send us your queries or feedback.



## Contact 📬
Feel free to reach out if you have any questions or feedback about PixelProbe!

- **Email**: najmi8815@gmail.com
---


--- 

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

