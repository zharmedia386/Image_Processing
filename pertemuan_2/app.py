from lzma import FILTER_LZMA2
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image 
import math

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

        # SLIDER HANDLER
        add_value = int(request.form["brightness_add"])
        # print(add_value)

		# WITH OPEN CV
        brightness_add_image_cv = cv2.add(original_image, add_value)
        cv2.imwrite("static/saved/with_opencv/brightness_add/" + filename, brightness_add_image_cv)

		# WITHOUT OPEN CV
        temp_image = np.asarray(original_image).astype('uint16')
        temp_image = temp_image + add_value
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

        # SLIDER HANDLER
        subtract_value = int(request.form["brightness_subtraction"])

		# WITH OPEN CV
        brightness_subtraction_image_cv = cv2.subtract(original_image, subtract_value)
        brightness_subtraction_image_cv = np.clip(brightness_subtraction_image_cv, 0, 255)
        cv2.imwrite("static/saved/with_opencv/brightness_subtraction/" + filename, brightness_subtraction_image_cv)

		# WITHOUT OPEN CV
        temp_image = np.asarray(original_image).astype('uint16')
        temp_image = temp_image - subtract_value
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

        # SLIDER HANDLER
        multiply_value = int(request.form["brightness_multiplication"])

		# WITH OPEN CV
        brightness_multiplication_image_cv = cv2.multiply(original_image, multiply_value)
        brightness_multiplication_image_cv = np.clip(brightness_multiplication_image_cv, 0, 255)
        cv2.imwrite("static/saved/with_opencv/brightness_multiplication/" + filename, brightness_multiplication_image_cv)

		# WITHOUT OPEN CV
        temp_image = np.asarray(original_image).astype('uint16')
        temp_image = temp_image * multiply_value
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

        # SLIDER HANDLER
        divide_value = int(request.form["brightness_divide"])

		# WITH OPEN CV
        brightness_divide_image_cv = cv2.divide(original_image, divide_value)
        brightness_divide_image_cv = np.clip(brightness_divide_image_cv, 0, 255)
        cv2.imwrite("static/saved/with_opencv/brightness_divide/" + filename, brightness_divide_image_cv)

		# WITHOUT OPEN CV
        temp_image = np.asarray(original_image).astype('uint16')
        temp_image = temp_image / divide_value
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


#########################################################################################
# BLACK AND WHITE
#########################################################################################

@app.route('/black_and_white')
def black_and_white():
    return render_template('black_and_white.html')
 
@app.route('/black_and_white', methods=['POST'])
def upload_image_black_and_white():
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
        black_and_white_image_cv = np.asarray(original_image).astype('uint16')
        grayImage = cv2.cvtColor(black_and_white_image_cv, cv2.COLOR_BGR2GRAY)
        (thresh, bnw) = cv2.threshold(grayImage, 125, 255, cv2.THRESH_BINARY)
        new_image = bnw.astype('uint8')
        cv2.imwrite("static/saved/with_opencv/black_and_white/" + filename, new_image)

        return render_template('black_and_white.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_black_and_white/<filename>')
def display_black_and_white(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_black_and_white_with_open_cv/black_and_white/<filename>')
def display_black_and_white_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/black_and_white/' + filename), code=301)

#########################################################################################
# BITWISE AND
#########################################################################################

@app.route('/bitwise_and')
def bitwise_and():
    return render_template('bitwise_and.html')
 
@app.route('/bitwise_and', methods=['POST'])
def upload_image_bitwise_and():
    if 'file1' not in request.files and 'file2' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' and file2.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if (file1 and allowed_file(file1.filename)) and (file2 and allowed_file(file2.filename)):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
		
        image_path1 = "static/uploads/" + filename1
        image_path2 = "static/uploads/" + filename2
        original_image1 = cv2.imread(image_path1)
        original_image2 = cv2.imread(image_path2)

		# WITH OPEN CV
        bitwise_and_image = cv2.bitwise_and(original_image1, original_image2)
        cv2.imwrite("static/saved/with_opencv/bitwise_and/" + filename1, bitwise_and_image)

        return render_template('bitwise_and.html', filename1=filename1, filename2=filename2)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_bitwise_and1/<filename>')
def display_bitwise_and1(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_bitwise_and2/<filename>')
def display_bitwise_and2(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_bitwise_and_with_open_cv/bitwise_and/<filename>')
def display_bitwise_and_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/bitwise_and/' + filename), code=301)


#########################################################################################
# BITWISE OR
#########################################################################################

@app.route('/bitwise_or')
def bitwise_or():
    return render_template('bitwise_or.html')
 
@app.route('/bitwise_or', methods=['POST'])
def upload_image_bitwise_or():
    if 'file1' not in request.files and 'file2' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' and file2.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if (file1 and allowed_file(file1.filename)) and (file2 and allowed_file(file2.filename)):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
		
        image_path1 = "static/uploads/" + filename1
        image_path2 = "static/uploads/" + filename2
        original_image1 = cv2.imread(image_path1)
        original_image2 = cv2.imread(image_path2)

		# WITH OPEN CV
        bitwise_or_image = cv2.bitwise_or(original_image1, original_image2)
        cv2.imwrite("static/saved/with_opencv/bitwise_or/" + filename1, bitwise_or_image)

        return render_template('bitwise_or.html', filename1=filename1, filename2=filename2)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_bitwise_or1/<filename>')
def display_bitwise_or1(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_bitwise_or2/<filename>')
def display_bitwise_or2(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_bitwise_or_with_open_cv/bitwise_or/<filename>')
def display_bitwise_or_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/bitwise_or/' + filename), code=301)


#########################################################################################
# BITWISE XOR
#########################################################################################

@app.route('/bitwise_xor')
def bitwise_xor():
    return render_template('bitwise_xor.html')
 
@app.route('/bitwise_xor', methods=['POST'])
def upload_image_bitwise_xor():
    if 'file1' not in request.files and 'file2' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' and file2.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if (file1 and allowed_file(file1.filename)) and (file2 and allowed_file(file2.filename)):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
		
        image_path1 = "static/uploads/" + filename1
        image_path2 = "static/uploads/" + filename2
        original_image1 = cv2.imread(image_path1)
        original_image2 = cv2.imread(image_path2)

		# WITH OPEN CV
        bitwise_xor_image = cv2.bitwise_xor(original_image1, original_image2)
        cv2.imwrite("static/saved/with_opencv/bitwise_xor/" + filename1, bitwise_xor_image)

        return render_template('bitwise_xor.html', filename1=filename1, filename2=filename2)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_bitwise_xor1/<filename>')
def display_bitwise_xor1(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_bitwise_xor2/<filename>')
def display_bitwise_xor2(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_bitwise_xor_with_open_cv/bitwise_xor/<filename>')
def display_bitwise_xor_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/bitwise_xor/' + filename), code=301)


#########################################################################################
# BITWISE NOT
#########################################################################################

@app.route('/bitwise_not')
def bitwise_not():
    return render_template('bitwise_not.html')
 
@app.route('/bitwise_not', methods=['POST'])
def upload_image_bitwise_not():
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
        bitwise_not_image = cv2.bitwise_not(original_image)
        cv2.imwrite("static/saved/with_opencv/bitwise_not/" + filename, bitwise_not_image)

        return render_template('bitwise_not.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_bitwise_not/<filename>')
def display_bitwise_not(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_bitwise_not_with_open_cv/bitwise_not/<filename>')
def display_bitwise_not_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/bitwise_not/' + filename), code=301)


#########################################################################################
# RANDOM
#########################################################################################

@app.route('/random')
def random():
    return render_template('random.html')
 
@app.route('/random', methods=['POST'])
def upload_image_random():
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

        # SLIDER HANDLER
        add_value = int(request.form["brightness_add"])
        subtract_value = int(request.form["brightness_subtraction"])
        multiply_value = int(request.form["brightness_multiplication"])
        divide_value = int(request.form["brightness_divide"])

		# WITH OPEN CV
        brightness_add_image_cv = cv2.add(original_image, add_value)

        brightness_subtraction_image_cv = cv2.subtract(brightness_add_image_cv, subtract_value)
        brightness_subtraction_image_cv = np.clip(brightness_subtraction_image_cv, 0, 255)

        brightness_multiplication_image_cv = cv2.multiply(brightness_subtraction_image_cv, multiply_value)
        brightness_multiplication_image_cv = np.clip(brightness_multiplication_image_cv, 0, 255)

        brightness_divide_image_cv = cv2.divide(brightness_multiplication_image_cv, divide_value)
        brightness_divide_image_cv = np.clip(brightness_divide_image_cv, 0, 255)
        
        cv2.imwrite("static/saved/with_opencv/random/" + filename, brightness_divide_image_cv)

        return render_template('random.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 

##########################################################################################
# GLOBAL THRESHOLDING
#########################################################################################

@app.route('/global_thresholding')
def global_thresholding():
    return render_template('global_thresholding.html')
 
@app.route('/global_thresholding', methods=['POST'])
def upload_image_global_thresholding():
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

        # Convert BGR to Grayscale Format
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 

		# WITH OPEN CV
        ret, th1 = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("static/saved/with_opencv/global_thresholding/" + filename, th1)

        return render_template('global_thresholding.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_global_thresholding/<filename>')
def display_global_thresholding(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_global_thresholding_with_open_cv/global_thresholding/<filename>')
def display_global_thresholding_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/global_thresholding/' + filename), code=301)


##########################################################################################
# ADAPTIVE MEAN THRESHOLDING
#########################################################################################

@app.route('/adaptive_mean_thresholding')
def adaptive_mean_thresholding():
    return render_template('adaptive_mean_thresholding.html')
 
@app.route('/adaptive_mean_thresholding', methods=['POST'])
def upload_image_adaptive_mean_thresholding():
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

        # Convert BGR to Grayscale Format
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 

		# WITH OPEN CV
        th2 = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite("static/saved/with_opencv/adaptive_mean_thresholding/" + filename, th2)

        return render_template('adaptive_mean_thresholding.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_adaptive_mean_thresholding/<filename>')
def display_adaptive_mean_thresholding(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_adaptive_mean_thresholding_with_open_cv/adaptive_mean_thresholding/<filename>')
def display_adaptive_mean_thresholding_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/adaptive_mean_thresholding/' + filename), code=301)


##########################################################################################
# ADAPTIVE GAUSSIAN THRESHOLDING
#########################################################################################

@app.route('/adaptive_gaussian_thresholding')
def adaptive_gaussian_thresholding():
    return render_template('adaptive_gaussian_thresholding.html')
 
@app.route('/adaptive_gaussian_thresholding', methods=['POST'])
def upload_image_adaptive_gaussian_thresholding():
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

        # Convert BGR to Grayscale Format
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 

		# WITH OPEN CV
        th3 = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite("static/saved/with_opencv/adaptive_gaussian_thresholding/" + filename, th3)

        return render_template('adaptive_gaussian_thresholding.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_adaptive_gaussian_thresholding/<filename>')
def display_adaptive_gaussian_thresholding(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_adaptive_gaussian_thresholding_with_open_cv/adaptive_gaussian_thresholding/<filename>')
def display_adaptive_gaussian_thresholding_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/adaptive_gaussian_thresholding/' + filename), code=301)


##########################################################################################
# OTSU THRESHOLDING
#########################################################################################

@app.route('/otsu_thresholding')
def otsu_thresholding():
    return render_template('otsu_thresholding.html')
 
@app.route('/otsu_thresholding', methods=['POST'])
def upload_image_otsu_thresholding():
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

        # Convert BGR to Grayscale Format
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 

		# WITH OPEN CV
        blur = cv2.GaussianBlur(original_image, (5,5), 0) # GaussianBlur Parameter: img -> source image; (5,5) -> Kernel Size . (height, width); 0 -> borderType
        ret2, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite("static/saved/with_opencv/otsu_thresholding/" + filename, thresh)

        return render_template('otsu_thresholding.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_otsu_thresholding/<filename>')
def display_otsu_thresholding(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_otsu_thresholding_with_open_cv/otsu_thresholding/<filename>')
def display_otsu_thresholding_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/otsu_thresholding/' + filename), code=301)


##########################################################################################
# HISTOGRAM EQUALIZATION
#########################################################################################

@app.route('/histogram_equalization')
def histogram_equalization():
    return render_template('histogram_equalization.html')
 
@app.route('/histogram_equalization', methods=['POST'])
def upload_image_histogram_equalization():
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

        # Convert BGR to Grayscale Format
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 

		# WITH OPEN CV
        # Spreads out the pixel intensity values
        equ = cv2.equalizeHist(original_image)

        # Configuration for displaying the result
        equalized_image = cv2.cvtColor(equ, cv2.COLOR_BGR2RGB)

        cv2.imwrite("static/saved/with_opencv/histogram_equalization/" + filename, equalized_image)

        return render_template('histogram_equalization.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_histogram_equalization/<filename>')
def display_histogram_equalization(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_histogram_equalization_with_open_cv/histogram_equalization/<filename>')
def display_histogram_equalization_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/histogram_equalization/' + filename), code=301)


##########################################################################################
# GAMMA CORRECTION
#########################################################################################

@app.route('/gamma_correction')
def gamma_correction():
    return render_template('gamma_correction.html')
 
@app.route('/gamma_correction', methods=['POST'])
def upload_image_gamma_correction():
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

        # Convert BGR to Grayscale Format
        grayscaled_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 

		# WITH OPEN CV
        # Compute Gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.mean(grayscaled_original_image)
        gamma = math.log(mid*255) / math.log(mean)
        # print(gamma)

        # Do Gamma correction
        gamma_correction_image = np.power(original_image, gamma).clip(0,255).astype(np.uint8)

        cv2.imwrite("static/saved/with_opencv/gamma_correction/" + filename, gamma_correction_image)

        return render_template('gamma_correction.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_gamma_correction/<filename>')
def display_gamma_correction(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_gamma_correction_with_open_cv/gamma_correction/<filename>')
def display_gamma_correction_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/gamma_correction/' + filename), code=301)


##########################################################################################
# LOW PASS FILTER
#########################################################################################

@app.route('/low_pass_filter')
def low_pass_filter():
    return render_template('low_pass_filter.html')
 
@app.route('/low_pass_filter', methods=['POST'])
def upload_image_low_pass_filter():
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

        # create the low pass filter
        lowFilter = np.ones((3,3),np.float32) / 9

		# WITH OPEN CV
        # apply the low pass filter to the image
        lowFilterImage = cv2.filter2D(original_image,-1,lowFilter)

        cv2.imwrite("static/saved/with_opencv/low_pass_filter/" + filename, lowFilterImage)

        return render_template('low_pass_filter.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_low_pass_filter/<filename>')
def display_low_pass_filter(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_low_pass_filter_with_open_cv/low_pass_filter/<filename>')
def display_low_pass_filter_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/low_pass_filter/' + filename), code=301)


##########################################################################################
# HIGH PASS FILTER
#########################################################################################

@app.route('/high_pass_filter')
def high_pass_filter():
    return render_template('high_pass_filter.html')
 
@app.route('/high_pass_filter', methods=['POST'])
def upload_image_high_pass_filter():
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

        # create the high pass filter
        highFilter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

		# WITH OPEN CV
        # apply the high pass filter to the image
        highFilterImage = cv2.filter2D(original_image,-1,highFilter)

        cv2.imwrite("static/saved/with_opencv/high_pass_filter/" + filename, highFilterImage)

        return render_template('high_pass_filter.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_high_pass_filter/<filename>')
def display_high_pass_filter(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_high_pass_filter_with_open_cv/high_pass_filter/<filename>')
def display_high_pass_filter_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/high_pass_filter/' + filename), code=301)



##########################################################################################
# BAND PASS FILTER
#########################################################################################

@app.route('/band_pass_filter')
def band_pass_filter():
    return render_template('band_pass_filter.html')
 
@app.route('/band_pass_filter', methods=['POST'])
def upload_image_band_pass_filter():
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

        # create the band pass filter
        bandFilter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

		# WITH OPEN CV
        # apply the band pass filter to the image
        bandFilterImage = cv2.filter2D(original_image,-1,bandFilter)


        cv2.imwrite("static/saved/with_opencv/band_pass_filter/" + filename, bandFilterImage)

        return render_template('band_pass_filter.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_band_pass_filter/<filename>')
def display_band_pass_filter(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_band_pass_filter_with_open_cv/band_pass_filter/<filename>')
def display_band_pass_filter_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/band_pass_filter/' + filename), code=301)



##########################################################################################
# SALT AND PEPPER NOISE
#########################################################################################

@app.route('/salt_and_pepper_noise')
def salt_and_pepper_noise():
    return render_template('salt_and_pepper_noise.html')
 
@app.route('/salt_and_pepper_noise', methods=['POST'])
def upload_image_salt_and_pepper_noise():
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

        # convert to grayscale
        # gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

		# WITH OPEN CV
        # convert to float32
        # gray = np.float32(gray)
        original_image = np.float32(original_image)

        # create the salt noise
        salt = np.random.randint(0,100,original_image.shape)
        salt = np.float32(salt)

        # add the salt noise
        saltImage = cv2.add(original_image,salt)


        cv2.imwrite("static/saved/with_opencv/salt_and_pepper_noise/" + filename, saltImage)

        return render_template('salt_and_pepper_noise.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display_salt_and_pepper_noise/<filename>')
def display_salt_and_pepper_noise(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_salt_and_pepper_noise_with_open_cv/salt_and_pepper_noise/<filename>')
def display_salt_and_pepper_noise_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/salt_and_pepper_noise/' + filename), code=301)


@app.route('/display_random/<filename>')
def display_random(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_random_with_open_cv/random/<filename>')
def display_random_with_open_cv(filename):
    #print('display_with_open_cv filename: ' + filename)
    return redirect(url_for('static', filename='saved/with_opencv/random/' + filename), code=301)

if __name__ == "__main__":
    app.run()