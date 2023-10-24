# Style My Picture


This web application is built using Flask, Python, and TensorFlow. It utilizes the Neural Style Transfer method, which is a generative approach to blend the content of an image with the style of another image.

## How it works

1. **Browse and Upload**: The user can click on the "Browse" button and choose an image of their choice.
2. **Generate Art**: After selecting an image, the user can click on "Show me the art" to apply the style transfer.
3. **Download**: The user can then right-click on the generated image and select "Download" to save it.

![Showmetheart](image_for_doc/Showmetheart.gif)

## Neural Style Transfer

Neural Style Transfer is a generative method that aims to transform the content of an image while retaining the style of another image. It uses deep learning techniques to achieve this, specifically, Convolutional Neural Networks (CNNs). It's an exciting way to blend images and create beautiful, artistic renditions.

## Installation

To run this application on your local machine, follow these steps:

1. Clone the repository from GitHub.
2. Navigate to the project directory and run `pip install -r requirements.txt` to install the required dependencies.
3. Run the `wsgi.py` file using Python.

## Note

For faster processing, the application stores images in RAM instead of saving them to disk when received by the server.

## Example Usage

After running the application, open your web browser and navigate to `http://127.0.0.1:8000/`. You should see the user interface of the application, where you can upload images and apply the Neural Style Transfer to generate artistic images.

## Reference

For more information on Neural Style Transfer, please refer to this [tutorial](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/style_transfer.ipynb#scrollTo=6msVLevwcRhm).

