import os
from skimage import io
import skimage.io as io
import io as python_io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import requests
from io import BytesIO
from mtcnn.mtcnn import MTCNN
import cv2


app = Flask(__name__)

def find_bounding_box(box):
    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]
    return left,top,right, bottom


#call mtcnn model constructor
net = MTCNN()

@app.route('/')
def welcome():
    return('''
    --------WELCOME TO FACE DETECTION MICROSERVICE--------------------------------------------------------------
    |                                                                                                          |
    | TRY CURL POST REQUEST TO USE FACE-DETECTION SERVICE:                                                     |
    | curl http://localhost:8000/face-detection -F "file=@image.jpg" -F "show_anonymize_image=True" |
    ------------------------------------------------------------------------------------------------------------
    ''')

@app.route('/face-detection', methods=['POST'])
def post():
    print('''

          ==============================================================================
          |                                                                             |
          |            Welcome to Face Detection Microservice                           |
          |                                                                             |
          ==============================================================================
         ''')
    print('''
          -------------------------------------------------------------------------------
          | --> Face-Detection Microservice returns bounding box of                     |
          |     detected faces for a given image                                        |
          | --> It is using MTCNN Model for detecting faces                             |
          | --> @param: show_anonymize_image --> Display anonymized faces in given image|
          | --> API:  http://localhost:8000/face-detection                              |
          | --> Request Parameter:                                                      |
          |        - file: file object (Required)                                       |
          |        - show_anonymize_image: display anonymized faces in image (Optional) |
          -------------------------------------------------------------------------------
         ''')
    print("Processing Face Detection API request")
    if request.content_type.startswith('multipart/form-data'):
        request_data = request.form.to_dict()

        errors = []
        if 'file' not in request.files:
            errors.append({'file': 'please provide file'})

        file_to_predict = request.files['file']

    else:
        return jsonify({'errors': [{'content_type': 'invalid request content_type'}]}, 400)

    show_anonymize_image = request_data.get('show_anonymize_image', False)
    result = {"predictions": []}

    img = io.imread(file_to_predict)
    #covert rgb to bgr for opencv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #detect_faces class from MTCNN model file
    dets = net.detect_faces(img)

    face_count = len(dets)

    print("[INFO] loading MTCNN Model ")
    print("[INFO] Number of faces detected: {}".format(face_count))


    # loop through all detections to anonymize faces
    if show_anonymize_image:
       for i in range(0, len(dets)):
           box = dets[i]['box']
           confidence = dets[i]['confidence']

           if confidence > 0.5:
              cv2.rectangle(img,(box[0], box[1]),
				  (box[0]+box[2], box[1] + box[3]), (219,165,55), -1)

       cv2.imshow("Face Anonymization Output",img)
       cv2.imwrite("anonymized_output.jpg",img)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
       result['predictions'].append({'success':'ok','faces': len(dets)})
       return jsonify(result)

    else:
       # loop through all detections to find bounding boxes
       for i in range(0, len(dets)):
           box = dets[i]['box']
           confidence = dets[i]['confidence']
           accuracy = "{:.2f}%".format(confidence * 100)
           if confidence > 0.5:
              left, top, right, bottom = find_bounding_box(box)

              print("[INFO] Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, left, top, right, bottom))

              result['predictions'].append(
                 {'box': {"xmin": left, "ymin": top, "xmax": right, "ymax": bottom},'accuracy': accuracy})

       return jsonify(result)


CORS(app, supports_credentials=True, allow_headers=['Content-Type', 'X-ACCESS_TOKEN', 'Authorization'])
application = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
