from tensorflow.python import keras
import streamlit as st
from PIL import Image, ImageDraw
import hilber_curve
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import math

class _Color:
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
class ColorEntropy(_Color):
    def getPoint(self, x):
        e = entropy(self.data, 32, x, len(self.symbol_map))
        # http://www.wolframalpha.com/input/?i=plot+%284%28x-0.5%29-4%28x-0.5%29**2%29**4+from+0.5+to+1
        def curve(v):
            f = (4*v - 4*v**2)**4
            f = max(f, 0)
            return f
        r = curve(e-0.5) if e > 0.5 else 0
        b = e**2
        return [
            int(255*r),
            0,
            int(255*b)
        ]

def drawmap_unrolled( size, csource, name):
    
    map = fs(2, size**2)
    c = Image.new("RGB", (size, size*4))

    cd = ImageDraw.Draw(c)
    step = len(csource)/float(len(map)*4)

    sofar = 0
    for quad in range(4):
        for i, p in enumerate(map):
            off = (i + (quad * size**2))
            color = csource.point(
                        int(off * step)
                    )
            x, y = tuple(p)
            cd.point(
                (x, y + (size * quad)),
                fill=tuple(color)
            )
    c =c.resize((224,224))
    pred = image.img_to_array(c)
    pred = np.array([pred])
    
    return c,pred
    

def fs(dim,size):
    return hilber_curve.Hilbert.fromSize(dim,size)


def entropy(data, blocksize, offset, symbols=256):
    
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
    entropy = 0
    for i in hist.values():
        p = i/float(blocksize)
        
        entropy += (p * math.log(p, base))
    return -entropy

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

st.markdown('<p style="font-family:Courier; color:White; font-size: 18px;text-align:center">This tool uses Hilbert space-filling curve with an entropy algorithm to create a binary visualization of .APK file, then it is passed to the CNN trained model which predicts whether a file is malware or legitimate application</p>',unsafe_allow_html=True)



st.markdown(footer_markdown(),unsafe_allow_html=True)




loaded_model = tf.keras.models.load_model("model.h5")

file = st.file_uploader('',type=['apk'])
st.markdown('<p style="font-family:Courier; color:White; font-size: 20px;text-align:center">Just place your file above and let\'s begin checking</p>',unsafe_allow_html=True)

col1, col2,col3 = st.columns(3)






if file is not None:
    source = file.getvalue()
    dst = "testing.png"

    calc = ColorEntropy(source)
    
    with st.spinner("Generating binary visualization, please wait..."):
        
        im,pred = drawmap_unrolled( 256, calc, dst)
        col2.markdown('<p style="font-family:Courier; color:White; font-size: 8px;text-align:left">Binary Visualization with Entropy algorithm </p>',unsafe_allow_html=True)
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
            

    
    
    
    
