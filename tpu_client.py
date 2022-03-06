import cv2, requests, socketio, time, base64, argparse, os
from pycoral.adapters import classify,common
from pycoral.utils.edgetpu import make_interpreter
import pandas as pd
import numpy as np
import asyncio
from collections import defaultdict

url = 'https://pklot.herokuapp.com'
#For sending to server

sio = socketio.Client()

@sio.event
def connect():
    print('[INFO] Successfully connected to server.')


@sio.event
def connect_error(data):
    print('[INFO] Failed to connect to server.')
    print(data)


@sio.event
def disconnect():
    print('[INFO] Disconnected from server.')

def convert_to_jpeg(image):
    frame = cv2.imencode('.jpg', image)[1].tobytes()
    frame = base64.b64encode(frame).decode('utf-8')
    return "data:image/jpeg;base64,{}".format(frame)


def rescale(val):
        return int(round(val * 0.3858))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    parser.add_argument(
        '-c', '--camera', type=int, default=1, 
        help="Which camera to pick")
    parser.add_argument(
        '-d', '--duration', type=int, default=3,
        help="Duration of how long each picture will stay"
    )
    print('[INFO] Connecting to server https://{}:...'.format(
            url, 5001))
    #connect_serv()
    sio.connect(
            '{}'.format(url),
            transports=['websocket'],
            namespaces=['/cv'])
    sio.sleep(1)
    args = parser.parse_args()
    interpreter = make_interpreter(*args.model.split('@'))
    interpreter.allocate_tensors()

    if common.input_details(interpreter, 'dtype') != np.uint8:
        raise ValueError('Only support uint8 input type.')

    cam1 = pd.read_csv("cam_data/camera{}.csv".format(args.camera))
    camera_photos = os.path.join( '2015-11-27/camera{}'.format(args.camera))
    pics = defaultdict(list)
    dims = defaultdict(list)
    c = 0
    originals = []

    params = common.input_details(interpreter, 'quantization_parameters')
    scale = params['scales']
    zero_point = params['zero_points']
    mean = 128.0
    std = 128.0
    file_names = []

    for f in os.listdir(camera_photos):
        img = os.path.join(camera_photos,f)
        img = cv2.imread(img)
        originals.append(img)
        rowid = []
        for i,row in cam1.iterrows():
            x1 = rescale(row['X'])
            y1 = rescale(row['Y'])
            x2 = x1 + rescale(row['W'])
            y2 = y1 + rescale(row['H'])
            crop_img = img[y1:y2,x1:x2]
            crop_img = cv2.resize(crop_img, (224,224), interpolation=cv2.INTER_CUBIC)
            pics[c].append(crop_img)
            dims[c].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2':y2})
            rowid.append(str(row['SlotId']))
        c=c+1
        file_names.append(str(f))

    images = np.array([np.array(pics[i]) for i in pics])

    c = 0
    for i in range(len(images)):
        print("predicting for image",i,"...")
        preds = []
        start = time.perf_counter()
        for j in range(len(images[i])):
            #if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
            common.set_input(interpreter, images[i][j])
            #else:
            #    normalized_input = (np.asarray(images[i][j]) - mean) / (std * scale) + zero_point
            #    np.clip(normalized_input, 0, 255, out=normalized_input)
            #    common.set_input(interpreter, normalized_input.astype(np.uint8))
            interpreter.invoke()
            classes = classify.get_classes(interpreter,2)
            #print(classes)
            #prediction = classes[0].id
            #preds.append(prediction)
            preds.append(str(classes[1].id))
        inference_time = time.perf_counter() - start
        print('Image time: %.1fms' % (inference_time * 1000))

        img = originals[i]
        # parking_data = defaultdict(list)
        # preds = [i.item() for i in preds]
        # print(preds)
        # print((preds[0]))
        for j in range(len(preds)):
            x1 = dims[c][j]['x1']
            y1 = dims[c][j]['y1']
            x2 = dims[c][j]['x2']
            y2 = dims[c][j]['y2']
            img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255,), 1) if int(preds[j]) else cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0), 2)
            cv2.putText(img, rowid[j], (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2,cv2.LINE_AA)
        parking_data = list(zip(rowid,preds))
        #print(parking_data)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        frame = base64.b64encode(frame).decode('utf-8')
        frame = "data:image/jpeg;base64,{}".format(frame)
        c=c+1
        #img = convert_to_jpeg(img)
        time.sleep(1)
        print("emitting",file_names[i])
        sio.emit('cvdata', {'image':frame},namespace='/cv')
        sio.emit('parkdata', {'lots': parking_data,'file': file_names[i]},namespace='/cv')
        print("emitted")
        time.sleep(args.duration)
    sio.disconnect()

if __name__ == "__main__":
    main()
