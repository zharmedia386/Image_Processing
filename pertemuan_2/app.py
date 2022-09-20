from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image 

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

#########################################################################################
# GRAYSCALE
#########################################################################################

@app.route('/grayscale')
def grayscale():
    return render_template('grayscale.html')
 
@app.route('/grayscale', methods=['POST'])
def upload_image_grayscale():
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
        cv2.imwrite("static/saved/with_opencv/grayscale/" + filename, gray_image)

		# WITHOUT OPEN CV
        RGB_image = cv2.resize(original_image, (500, 500)) 
        blue = RGB_image[:,:,0]
        green = RGB_image[:,:,1]
        red = RGB_image[:,:,2]

        grayscale = blue/3 + green/3 + red/3

        RGB_image[:,:,0] = grayscale
        RGB_image[:,:,1] = grayscale
        RGB_image[:,:,2] = grayscale
        cv2.imwrite("static/saved/without_opencv/grayscale/" + filename, RGB_image)

        return render_template('grayscale.html', filename=filename, blue=blue, green=green, red=red, grayscale=grayscale)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_grayscale/<filename>')
def display_grayscale(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_grayscale_with_open_cv/grayscale/<filename>')
def display_grayscale_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/grayscale/' + filename), code=301)

@app.route('/display_grayscale_without_open_cv/grayscale/<filename>')
def display_grayscale_without_open_cv(filename):
    #print('display_without_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/without_opencv/grayscale/' + filename), code=301)


#########################################################################################
# INVERS
#########################################################################################

@app.route('/invers')
def invers():
    return render_template('invers.html')
 
@app.route('/invers', methods=['POST'])
def upload_image_invers():
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
        invers_image = cv2.bitwise_not(original_image)
        cv2.imwrite("static/saved/with_opencv/invers/" + filename, invers_image)

		# WITHOUT OPEN CV
        RGB_image = cv2.resize(original_image, (500, 500)) 

        blue = RGB_image[:,:,0]
        green = RGB_image[:,:,1]
        red = RGB_image[:,:,2]

        # print(blue, green, red)

        RGB_image[:,:,0] = 255 - blue
        RGB_image[:,:,1] = 255 - green
        RGB_image[:,:,2] = 255 - red
        cv2.imwrite("static/saved/without_opencv/invers/" + filename, RGB_image)

        return render_template('invers.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_invers/<filename>')
def display_invers(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_invers_with_open_cv/invers/<filename>')
def display_invers_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/invers/' + filename), code=301)

@app.route('/display_invers_without_open_cv/invers/<filename>')
def display_invers_without_open_cv(filename):
    #print('display_without_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/without_opencv/invers/' + filename), code=301)


#########################################################################################
# CROPPING
#########################################################################################

@app.route('/cropping')
def cropping():
    return render_template('cropping.html')
 
@app.route('/cropping', methods=['POST'])
def upload_image_cropping():
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
        cropped_image  = original_image[80:150, 150:230]
        cv2.imwrite("static/saved/with_opencv/cropping/" + filename, cropped_image)

        return render_template('cropping.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_cropping/<filename>')
def display_cropping(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_cropping_with_open_cv/cropping/<filename>')
def display_cropping_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/cropping/' + filename), code=301)

@app.route('/display_cropping_without_open_cv/cropping/<filename>')
def display_cropping_without_open_cv(filename):
    #print('display_without_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/without_opencv/cropping/' + filename), code=301)


#########################################################################################
# BRIGHTNESS ADD 
#########################################################################################

@app.route('/brightness_add')
def brightness_add():
    return render_template('brightness_add.html')
 
@app.route('/brightness_add', methods=['POST'])
def upload_image_brightness_add():
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
        brightness_add_image_cv = cv2.add(original_image, 100)
        cv2.imwrite("static/saved/with_opencv/brightness_add/" + filename, brightness_add_image_cv)

		# WITHOUT OPEN CV
        temp_image = np.asarray(original_image).astype('uint16')
        temp_image = temp_image + 100
        temp_image = np.clip(temp_image, 0, 255)
        brightness_add_image_without_cv = temp_image.astype('uint8')
        cv2.imwrite("static/saved/without_opencv/brightness_add/" + filename, brightness_add_image_without_cv)

        return render_template('brightness_add.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_brightness_add/<filename>')
def display_brightness_add(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_brightness_add_with_open_cv/brightness_add/<filename>')
def display_brightness_add_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/brightness_add/' + filename), code=301)

@app.route('/display_brightness_add_without_open_cv/brightness_add/<filename>')
def display_brightness_add_without_open_cv(filename):
    #print('display_without_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/without_opencv/brightness_add/' + filename), code=301)

#########################################################################################
# BRIGHTNESS SUBTRACTION 
#########################################################################################

@app.route('/brightness_subtraction')
def brightness_subtraction():
    return render_template('brightness_subtraction.html')
 
@app.route('/brightness_subtraction', methods=['POST'])
def upload_image_brightness_subtraction():
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
        brightness_subtraction_image_cv = cv2.subtract(original_image, 100)
        brightness_subtraction_image_cv = np.clip(brightness_subtraction_image_cv, 0, 255)
        cv2.imwrite("static/saved/with_opencv/brightness_subtraction/" + filename, brightness_subtraction_image_cv)

		# WITHOUT OPEN CV
        temp_image = np.asarray(original_image).astype('uint16')
        temp_image = temp_image - 100
        temp_image = np.clip(temp_image, 0, 255)
        brightness_subtraction_image_without_cv = temp_image.astype('uint8')
        cv2.imwrite("static/saved/without_opencv/brightness_subtraction/" + filename, brightness_subtraction_image_without_cv)

        return render_template('brightness_subtraction.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_brightness_subtraction/<filename>')
def display_brightness_subtraction(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_brightness_subtraction_with_open_cv/brightness_subtraction/<filename>')
def display_brightness_subtraction_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/brightness_subtraction/' + filename), code=301)

@app.route('/display_brightness_subtraction_without_open_cv/brightness_subtraction/<filename>')
def display_brightness_subtraction_without_open_cv(filename):
    #print('display_without_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/without_opencv/brightness_subtraction/' + filename), code=301)

#########################################################################################
# BRIGHTNESS MULTIPLICATION 
#########################################################################################

@app.route('/brightness_multiplication')
def brightness_multiplication():
    return render_template('brightness_multiplication.html')
 
@app.route('/brightness_multiplication', methods=['POST'])
def upload_image_brightness_multiplication():
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
        brightness_multiplication_image_cv = cv2.multiply(original_image, 1.25)
        brightness_multiplication_image_cv = np.clip(brightness_multiplication_image_cv, 0, 255)
        cv2.imwrite("static/saved/with_opencv/brightness_multiplication/" + filename, brightness_multiplication_image_cv)

		# WITHOUT OPEN CV
        temp_image = np.asarray(original_image).astype('uint16')
        temp_image = temp_image * 1.25
        temp_image = np.clip(temp_image, 0, 255)
        brightness_multiplication_image_without_cv = temp_image.astype('uint8')
        cv2.imwrite("static/saved/without_opencv/brightness_multiplication/" + filename, brightness_multiplication_image_without_cv)

        return render_template('brightness_multiplication.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_brightness_multiplication/<filename>')
def display_brightness_multiplication(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_brightness_multiplication_with_open_cv/brightness_multiplication/<filename>')
def display_brightness_multiplication_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/brightness_multiplication/' + filename), code=301)

@app.route('/display_brightness_multiplication_without_open_cv/brightness_multiplication/<filename>')
def display_brightness_multiplication_without_open_cv(filename):
    #print('display_without_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/without_opencv/brightness_multiplication/' + filename), code=301)


#########################################################################################
# BRIGHTNESS DIVIDE 
#########################################################################################

@app.route('/brightness_divide')
def brightness_divide():
    return render_template('brightness_divide.html')
 
@app.route('/brightness_divide', methods=['POST'])
def upload_image_brightness_divide():
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
        brightness_divide_image_cv = cv2.divide(original_image, 2)
        brightness_divide_image_cv = np.clip(brightness_divide_image_cv, 0, 255)
        cv2.imwrite("static/saved/with_opencv/brightness_divide/" + filename, brightness_divide_image_cv)

		# WITHOUT OPEN CV
        temp_image = np.asarray(original_image).astype('uint16')
        temp_image = temp_image / 2
        temp_image = np.clip(temp_image, 0, 255)
        brightness_divide_image_without_cv = temp_image.astype('uint8')
        cv2.imwrite("static/saved/without_opencv/brightness_divide/" + filename, brightness_divide_image_without_cv)

        return render_template('brightness_divide.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_brightness_divide/<filename>')
def display_brightness_divide(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_brightness_divide_with_open_cv/brightness_divide/<filename>')
def display_brightness_divide_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/brightness_divide/' + filename), code=301)

@app.route('/display_brightness_divide_without_open_cv/brightness_divide/<filename>')
def display_brightness_divide_without_open_cv(filename):
    #print('display_without_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/without_opencv/brightness_divide/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)