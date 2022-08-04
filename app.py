#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage import io
from keras.models import load_model
import cv2
from PIL import Image #use PIL
import numpy as np


# In[2]:


app = Flask(__name__)


# In[3]:


@app.route("/", methods = ["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        print("File Received")
        filename = secure_filename(file.filename)
        print(filename)
        file.save("./static/"+filename)
        file = open("./static/"+filename, "r")
        model = load_model("Pneumonia")
        image = cv2.imread("./static/" + filename) # remove for cloud
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # remove for cloud
        img = cv2.merge([gray, gray, gray])
        
        img.resize((150, 150, 3))
        img = np.asarray(img, dtype="float32") #need to transfer to np to reshape
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) #rgb to reshape to 1,100,100,3
        pred = model.predict(img)
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="WAITING"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




