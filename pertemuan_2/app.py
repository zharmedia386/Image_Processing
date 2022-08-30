from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
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
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
		
        image_path = "static/uploads/" + filename
        original_image = cv2.imread(image_path)

		# WITH OPEN CV
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/saved/with_opencv/" + filename, gray_image)

		# WITHOUT OPEN CV
        RGB_image = cv2.resize(original_image, (500, 500)) 
        blue = RGB_image[:,:,0]
        green = RGB_image[:,:,1]
        red = RGB_image[:,:,2]

        grayscale = blue/3 + green/3 + red/3

        RGB_image[:,:,0] = grayscale
        RGB_image[:,:,1] = grayscale
        RGB_image[:,:,2] = grayscale
        cv2.imwrite("static/saved/without_opencv/" + filename, gray_image)

        return render_template('index.html', filename=filename, blue=blue, green=green, red=red, grayscale=grayscale)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_with_open_cv/<filename>')
def display_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/' + filename), code=301)

@app.route('/display_without_open_cv/<filename>')
def display_without_open_cv(filename):
    #print('display_without_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/without_opencv/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()