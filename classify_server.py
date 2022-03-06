from flask import Flask
from flask import render_template
from flask import Response
import cv2
import pandas as pd
import numpy as np
import os,time, argparse
from collections import defaultdict

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
# For locally running and displaying inference 

app = Flask(__name__)

def rescale(val):
        return int(round(val * 0.3858))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    # parser.labels', help='File path of labels file.')
    parser.add_argument(
        '-k', '--top_k', type=int, default=1,
        help='Max number of classification results')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.0,
        help='Classification score threshold')
    parser.add_argument(
        '-a', '--input_mean', type=float, default=128.0,
        help='Mean value for input normalization')
    parser.add_argument(
        '-s', '--input_std', type=float, default=128.0,
        help='STD value for input normalization')
    parser.add_argument(
        '-c', '--camera', type=int, default=1, help="Which camera to pick")
    parser.add_argument(
	'-d', '--duration', type=int, default=3, help="duration of how long each picture stays")
    args = parser.parse_args()

    interpreter = make_interpreter(*args.model.split('@'))
    interpreter.allocate_tensors()

    if common.input_details(interpreter, 'dtype') != np.uint8:
        raise ValueError('Only support uint8 input type.')

    size = common.input_size(interpreter)
    
    cam1 = pd.read_csv("cam_data/camera{}.csv".format(args.camera))
    camera_photos = os.path.join(os.path.dirname("2015-11-27"), '2015-11-27/camera{}'.format(args.camera))
    pics = defaultdict(list)
    c = 0
    originals = []
  
    for f in os.listdir(camera_photos):
        img = os.path.join(camera_photos,f)
        img = cv2.imread(img)
        originals.append(img)
        for i,row in cam1.iterrows():
            x1 = rescale(row['X'])
            y1 = rescale(row['Y'])
            x2 = x1 + rescale(row['W'])
            y2 = y1 + rescale(row['H'])
            crop_img = img[y1:y2,x1:x2]
            crop_img = cv2.resize(crop_img, (224,224),interpolation=cv2.INTER_CUBIC)
            pics[c].append(crop_img)
        c=c+1

        params = common.input_details(interpreter, 'quantization_parameters')
        scale = params['scales']
        zero_point = params['zero_points']
        mean = args.input_mean
        std = args.input_std
    
    images = np.array([np.array(pics[i]) for i in pics])
    batch_prediction = []

    for i in range(len(images)):
        print("predicting for image",i,"...")
        preds = []
        start = time.perf_counter()
        for j in range(len(images[i])):
            # if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
            common.set_input(interpreter, images[i][j])
            # el.set_input(interpreter, normalized_input.astype(np.uint8))
            interpreter.invoke()
            classes =  classify.get_classes(interpreter,1)
            preds.append(classes[0].id)
        inference_time = time.perf_counter() - start
        print('Image time: %.1fms' % (inference_time * 1000))

        img = originals[i]
        for j,row in cam1.iterrows():
            x1 = rescale(row['X'])
            y1 = rescale(row['Y'])
            x2 = x1 + rescale(row['W'])
            y2 = y1 + rescale(row['H'])
            img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0,), 2) if int(preds[j]) else cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255), 1)
            cv2.putText(img, str(row['SlotId']), (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2,cv2.LINE_AA)
        # cv2_imshow(img)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        print("posting img", i)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        preds = np.array(preds).flatten()
        batch_prediction.append(preds)
        time.sleep(args.duration)
        
        # print("prediction list",batch_prediction[i])
    quit()
    # j=0


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
  app.run(host='0.0.0.0')
