from fileinput import filename
from stylemypicture import app
import os
from flask import Flask, request, redirect, send_file, url_for, render_template, flash
from werkzeug.utils import secure_filename
#import numpy as np
#import cv2
import base64
import io
from PIL import Image
import gc
from stylemypicture.processor import *

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def PIL_to_HTML_display(PIL_img):
    '''
    using this function to avoid saving data on disk / for realtime  processing
    in python:
        return render_template("index.html", data=PIL_to_HTML_display(PIL_img))
    in html:
        <img src="data:image/jpeg;base64,{{data }}" alt="" width="480px" height="360px">
    '''
    binary_buffer = io.BytesIO()
    PIL_img.save(binary_buffer, "JPEG")
    encoded_img = base64.b64encode(binary_buffer.getvalue())
    return encoded_img.decode('utf-8')


style_image_list = ["im01", "im02", "im03", "im04", "im05", "im06", "im07", "im08",
                    "im09", "im10", "im11", "im12", "im13", "im14", "im15",
                    "im16", "im17", "im18", "im19", "im20", "im21"]


def process_my_picture(img_to_process, style_img_path):
    content_image = tensor_load_rgbimage(
        img_to_process, size=512, keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimage(style_img_path, size=512).unsqueeze(0)

    style = preprocess_batch(style)

    # _______________________Create MSG-Net and load pre-trained weights___________
    style_model = Net(ngf=128)
    #model_dict = torch.load('21styles.model')
    path_to_model = os.path.join('stylemypicture', 'static',
                                 'model', '21styles.model')
    model_dict = torch.load(path_to_model)
    #cleanup some keys in th model
    keys_to_clean_list=[]
    for key, value in model_dict.items():
        if key.endswith(('running_mean', 'running_var')):
            keys_to_clean_list.append(key)
    for key in keys_to_clean_list:
        del model_dict[key]
        gc.collect()
    del keys_to_clean_list
    gc.collect()
    style_model.load_state_dict(model_dict, False)
    del model_dict
    gc.collect()

    # __________Set the style target and generate outputs____________

    style_v = Variable(style)
    del style
    gc.collect()
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)
    output = style_model(content_image)
    del style_model
    gc.collect()
    return get_PIL_from_tensor(output.data[0], False)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            '''
                Convert image grom PIL to opencv
                I = numpy.asarray(PIL.Image.open('test.jpg'))
                Do some stuff to I, then, convert it back to an image:
                im = PIL.Image.fromarray(numpy.uint8(I))
            '''
            # read image from request without saving it on disk
            image = request.files['file'].read()
            image = io.BytesIO(image)

            #image = Image.open(image)

            img_style_number = request.form['slidenumber']

            path_to_choosed_style = os.path.join('stylemypicture', 'static', 'images',
                                                 str(style_image_list[int(img_style_number)]) + ".jpg")
            img = process_my_picture(image, path_to_choosed_style)

            data = {
                # "out": path,

                "processed_img": PIL_to_HTML_display(img)
                # "received_img": PIL_to_HTML_display(Image.open(image)),

            }
            return render_template("index.html", data=data)
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')
