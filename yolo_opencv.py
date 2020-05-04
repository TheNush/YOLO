import cv2
import argparse
import numpy as np
import time

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=False,
                help = 'path to input video')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

cap = cv2.VideoCapture(0)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

net = cv2.dnn.readNet(args.weights, args.config)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

start = time.time()
frame_count = 0

while (cap.isOpened()):
	ret, image = cap.read()
	frame_count += 1

	Width = image.shape[1]
	Height = image.shape[0]
	scale = 1.0/255.0
	# scale = 1.0

	blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(get_output_layers(net))

	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.5
	nms_threshold = 0.4
	count = 0


	for out in outs:
	    for detection in out:
	        scores = detection[5:]
	        class_id = np.argmax(scores)
	        confidence = scores[class_id]
	        if confidence > 0.5:
	            center_x = int(detection[0] * Width)
	            center_y = int(detection[1] * Height)
	            w = int(detection[2] * Width)
	            h = int(detection[3] * Height)
	            x = center_x - w / 2
	            y = center_y - h / 2
	            class_ids.append(class_id)
	            confidences.append(float(confidence))
	            boxes.append([x, y, w, h])
	            if str(classes[class_id]) == "person":
	            	count += 1


	indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

	for i in indices:
	    i = i[0]
	    box = boxes[i]
	    x = box[0]
	    y = box[1]
	    w = box[2]
	    h = box[3]
	    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

	cv2.imshow("object detection", image)
	if (frame_count == 30):
        	seconds = time.time() - start
        	print("FPS: ", 30/seconds)

        	frame_count = 0
        	start = time.time()

	# print("Number of People: ", count)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
	    
cap.release()
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
