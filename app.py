#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import acapture
import time
import numpy as np
import sys
import os.path

#Initialize the Flask app
app = Flask(__name__)

camera = acapture.open(0)

# Initialize the parameters
objectnessThreshold = 0.5 # Objectness threshold
confThreshold = 0.5      # Confidence threshold
nmsThreshold = 0.4        # Non-maximum suppression threshold
imgw = 416            # Width of network's input image
imgh = 416           # Height of network's input image


classes = ["kirkland water","fiji water"]

mConfig = "./yolo/yolov3-water.cfg"
mWeights = "./yolo/yolov3-water.weights"

yolonet = cv2.dnn.readNetFromDarknet(mConfig, mWeights)


# Get the names of the output layers
def getOutputsNames(yolonet):
    # Get the names of all the layers in the network
    layersNames = yolonet.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in yolonet.getUnconnectedOutLayers()]



# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    if classId == 1:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    else:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 20, 147), 3)
        
    label = '%.2f' % conf        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    return frame
    

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold :
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        frame = drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
    return frame



def get_video():  
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:       
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(frame, 1/255, (imgw, imgh), [0,0,0], 1, crop=False)
            yolonet.setInput(blob)
            outs = yolonet.forward(getOutputsNames(yolonet))
            frame = postprocess(frame, outs)
            ret, buffer = cv2.imencode('.jpg', frame)            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
            

         

@app.route('/video_feed')
def video_feed():
    return Response(get_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    global fps
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)