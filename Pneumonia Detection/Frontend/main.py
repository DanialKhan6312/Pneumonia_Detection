import os
from flask import Flask, render_template, request
app = Flask(__name__,static_url_path='/static')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/index')
def main():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/products')
def project():
    return render_template('products.html')
@app.route ('/ans' , methods = ['GET','POST'])
def pred():

    if request.method == 'POST':
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import load_model
        file = request.files['xray']
        file.filename = "upload.jpeg"
        file_name = file.filename
        base = os.path.dirname(__file__)
        filepath = os.path.join(base, 'uploads', file.filename)
        file.save(filepath)
    model = load_model("Model")

    data_gen= ImageDataGenerator(rescale=1/255).flow_from_directory(base,classes=["uploads"],class_mode=None, target_size=(100,100))
    predi = (model.predict(data_gen))
    message = str(round(predi[0][1]*100,2))+"%"
    return render_template("products.html",likelyhood = message)
if __name__ == '__main__':
   app.run(debug = True)
