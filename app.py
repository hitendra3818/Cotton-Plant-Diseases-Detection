import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask import Flask, render_template, request

# Load Model
model = load_model("model/cotton_plant_diseases.h5")
print('...........Model Loaded..........')

def model_output (path):
    raw_img = image.load_img(path,target_size=(150,150))   # importing image is converted to 64*64
    raw_img = image.img_to_array(raw_img) # convert image to array 
    raw_img = np.expand_dims(raw_img, axis=0)
    raw_img = raw_img/255                     # data max normalization
    Probability = model.predict(raw_img).round(3)  # probability
    plt.imshow(cv2.imread(path))                # Display Image
    print('Probability : ',Probability)
    pred = np.argmax(Probability)# get the index of max value
 
    if pred == 0:
        return "diseased cotton leaf" # if index 0 burned leaf
    elif pred == 1:
          return 'diseased cotton plant' # # if index 1
    elif pred == 2:
          return 'fresh cotton leaf'  # if index 2  fresh leaf
    else:
        return "fresh cotton plant" # if index 3
    
 ##>>...........pred_diseases..........END>>##

## Create Flask instance
app = Flask(__name__)
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred, output_page = model_output(cott_plant=file_path)
               
        return render_template(output_page, pred_output = pred, user_image = file_path)
     
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False)
 
