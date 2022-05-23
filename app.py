from tensorflow.python import keras
import streamlit as st
from PIL import Image, ImageDraw
import curve
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocentropiaeness_input
import math

class kolor:
    def __init__(self, data):
        self.data, self.block = data, None
        self.data = data
        s = list(set(data))
        s.sort()
        self.symbol_map = {v : i for (i, v) in enumerate(s)}

    def __len__(self):
        return len(self.data)
    def point(self, x):
        if self.block and (self.block[0]<=x<self.block[1]):
            return self.block[2]
        else:
            return self.getPoint(x)
class kolorujEntropie(kolor):
    def getPoint(self, x):
        e = entropia(self.data, 32, x, len(self.symbol_map))
        
        def krzywa(v):
            f = (4*v - 4*v**2)**4
            f = max(f, 0)
            return f
        r = krzywa(e-0.5) if e > 0.5 else 0
        b = e**2
        return [
            int(255*r),
            0,
            int(255*b)
        ]

def narysujMapeRozwinieta( size, csource, name):
    
    map = fs(2, size**2)
    c = Image.new("RGB", (size, size*4))

    d = ImageDraw.Draw(c)
    krok = len(csource)/float(len(map)*4)

    
    for k in range(4):
        for i, p in enumerate(map):
            off = (i + (k * size**2))
            kolor = csource.point(
                        int(off * krok)
                    )
            x, y = tuple(p)
            d.point(
                (x, y + (size * k)),
                fill=tuple(kolor)
            )
    c =c.resize((224,224))
    pred = image.img_to_array(c)
    pred = np.array([pred])
    
    return c,pred
    

def fs(dim,size):
    return curve.Hilbert.fromSize(dim,size)


def entropia(data, blocksize, offset, symbols=256):
    
    if offset < blocksize/2:
        start = 0
    elif offset > len(data)-blocksize/2:
        start = len(data)-blocksize/2
    else:
        start = offset-blocksize/2
    hist = {}
    for i in data[int(start):int(start+blocksize)]:
        hist[i] = hist.get(i, 0) + 1
    base = min(blocksize, symbols)
    entropia = 0
    for i in hist.values():
        p = i/float(blocksize)
        
        entropia += (p * math.log(p, base))
    return -entropia

def footer_markdown():
    footer="""
    <style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }
    
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }
    .reportview-container {
        background-color: #0F2536;
    }
    
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0F2536;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p style='display: block; text-align: center;font-family:Courier; color:White; font-size: 15px;'>Developed by <a style='display: block; text-align: center;font-family:Courier; color:White; font-size: 15px;' >Kacper Ratajczak</a></p>
    </div>
    """
    return footer


st.set_page_config(page_title='Android Malware Detection')

st.markdown('<h1 style="font-family:Courier; color:White; font-size: 30px;text-align:center">Android Malware Detection Tool</h1>',unsafe_allow_html=True)

st.markdown('<p style="font-family:Courier; color:White; font-size: 18px;text-align:center">This tool uses Hilbert space-filling curve with an entropia algorithm to create a binary visualization of .APK file, then it is passed to the CNN trained model which predicts whether a file is malware or legitimate application</p>',unsafe_allow_html=True)



st.markdown(footer_markdown(),unsafe_allow_html=True)




loaded_model = tf.keras.models.load_model("model.h5")

file = st.file_uploader('',type=['apk'])
st.markdown('<p style="font-family:Courier; color:White; font-size: 20px;text-align:center">Just place your file above and let\'s begin checking</p>',unsafe_allow_html=True)

col1, col2,col3 = st.columns(3)






if file is not None:
    source = file.getvalue()
    dst = "testing.png"

    calc = kolorujEntropie(source)
    
    with st.spinner("Generating binary visualization, please wait..."):
        
        im,pred = narysujMapeRozwinieta( 256, calc, dst)
        col2.markdown('<p style="font-family:Courier; color:White; font-size: 8px;text-align:left">Binary Visualization with entropia algorithm </p>',unsafe_allow_html=True)
        col2.image(im)
        

    
    with st.spinner("Making prediction, please wait..."):
            predictions = loaded_model.predict(pred)
    
            y_pred=np.argmax(predictions)
            if y_pred == 0:
                label = 'cerberus malware'
                percent = predictions[-1][0]
            elif y_pred == 1:
                label = 'hydra malware'
                percent = predictions[-1][1]
            elif y_pred == 2:
                label = 'alien malware'
                percent = predictions[-1][2]
            elif y_pred == 3:
                label = 'unclassified malware'
                percent = predictions[-1][3]
            else:
                label = 'harmless application'
                percent = predictions[-1][4]
            percent = answer = str(round(percent*100, 0))
            col2.markdown(f'<p style="font-family:Courier; color:White; font-size: 20px;text-align:center">file is {label} for {percent}% </p>',unsafe_allow_html=True)
            

    
    
    
    
