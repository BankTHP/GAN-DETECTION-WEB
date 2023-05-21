#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import torch
import pandas as pd

# gpu_id = torch.cuda.current_device()

app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD=True
)
 
UPLOAD_FOLDER = 'static/uploads/'
MODEL_DIR = 'model/Final_Model.h5'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def mainpage():
    return render_template('generate.html')
 
@app.route('/generate', methods=['POST'])
def upload_image():
    data = generateAttGAN('1',request.get_json()['attribute'])
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(filename)

        # return render_template('generate.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
    return redirect(request.url)


@app.route('/detect')
def detectpage():
    return render_template('detect.html')
 
@app.route('/detect', methods=['POST'])
def detect_upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        detect = detection(filename)
        return render_template('detect.html', filename=filename,fake=detect[0],percent=detect[1])
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)



@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def feature_extraction(img):
    model = ResNet50(weights='imagenet',include_top=False,pooling="avg")
    imgz = image.load_img(f'static/uploads/{img}', target_size=(224, 224))
    x = image.img_to_array(imgz)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)

    
    return features
    
def detection(filename) :
    feature = feature_extraction(filename)
    MLP = tf.keras.models.load_model(MODEL_DIR, compile=False)
    predict = MLP.predict(feature)
    result = 'Fake' if predict[0][0] > predict[0][1] else 'Real'
    
    return result,predict

def generateAttGAN(img,attribute,) :
    img = img_list_predict(img) 
    for i in attribute :

        command = (
            "python AttGAN/test_multi.py "
            "--experiment_name 384_shortcut1_inject1_none_hq "
            f"--test_atts {i} "
            "--test_ints 1 "
            "--custom_img "
            "--custom_data static/uploads " #path รูป เพราะทุกรูปอัพที่ upload อยู่แล้ว
            f"--custom_attr /content/drive/MyDrive/วิจัย/AttGAN/data/0/{i}.txt " #path text file
            "--gpu"
        )


        os.system(command)


    return #path ที่ generate มา

def img_list_predict(img):
    
    labels = ["Bald", "Bangs","Black_Hair", "Blond_Hair","Brown_Hair", "Bushy_Eyebrows","Eyeglasses","Male", "Mouth_Slightly_Open", "Mustache","No_Beard","Pale_Skin","Young"]
    data = {}
    command = (
            "python predict_attr/test.py "
            f"--gpu_ids {gpu_id} "
            "--model_type resnet50 "
            "--img_size 256 "
            "--float16 false "
            f"--img_path {img} "
            "--att_path predict_attr/data_list/att_map.txt "
            "--checkpoint_dir predict_attr/FAC_resnet50_AW_V1"
        )

    x = os.system(command) #ได้เป็น list กลับมา

    for line in x:
        if ':' in line:
            parts = line.split(':')
            label = parts[0].strip().replace('"', '').strip()
            if label in labels:
                value = float(parts[1].strip().replace(',', ''))
                data[label] = value

    for label, value in data.items():
        if value > 0.5:
            data[label] = 1
        else:
            data[label] = -1

    df = pd.DataFrame(data, index=[0])
    df.columns = labels
    df = df.set_index(pd.Index([img])).reset_index()
    df = df.rename(columns={'index': 0})
    df_to_list(df,img) 

def df_to_list(df,img): #ให้มันทำเป็น text file
    
    df.to_csv(f'text/{img}.txt',header=None,index=None, sep=' ', mode='a')
    with open(f'text/{img}.txt', 'r+') as file: 
        file_data = file.read() 
        file.seek(0, 0) #เขียนจำนวนคน
        file.write(str(len(df)) + '\n' + 'Bald Bangs Black_Hair Blond_Hair Brown_Hair Bushy_Eyebrows Eyeglasses Male Mouth_Slightly_Open Mustache No_Beard Pale_Skin Young' + '\n'+file_data)
    
    return 'text/{img}.txt'


    


if __name__ == "__main__":
    app.run(debug=True)