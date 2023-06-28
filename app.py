from flask import Flask, render_template, request, jsonify
import os
from deeplearning import object_detection

# webserver gateway interface
app = Flask(__name__)

UPLOAD_PATH = ('static/upload/')


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        text_list = object_detection(path_save, filename)

        print(text_list)

        return render_template('index.html', upload=True, upload_image=filename, text=text_list, no=len(text_list))

    return render_template('index.html', upload=False)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        upload_file = request.files['image']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        text_list = object_detection(path_save, filename)
        data_object = {
            'licensePlates': text_list
        }
        return jsonify(data_object)
    else:
        data_object = {
            "licensePlates": [
                "No image provided"
            ]
        }
        return jsonify(data_object)


if __name__ == "__main__":
    app.run(debug=True)
