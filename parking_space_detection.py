from flask import Flask, jsonify
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
import sys
import torch

app = Flask(__name__)
# CL argument 1
app.config['arg1'] = sys.argv[1]
# CL argument 2
app.config['arg2'] = sys.argv[2]

plt.rcParams["figure.figsize"] = [13.91, 9.51]
plt.rcParams["figure.autolayout"] = True

def loadJSONParkingSpaceData(pointMapHash):
    # Load the vertices from the JSON file
    parking_space_shapes_json = []
    with open('parking_space_config' + str(pointMapHash) + '.json', 'r') as f:
        parking_space_shapes_json = json.load(f)
    
    # Convert the json data into a multi-dimensional numpy array
    parking_spaces = []
    for parking_space in parking_space_shapes_json:
        parking_spaces.append(np.array(parking_space, np.int32))
        
    return parking_spaces
    
def createJSONParkingSpaceData(pointMapHash, parking_spaces):
    # Convert the vertices to a JSON serializable format
    shapes_json = []
    for shape in parking_spaces:
        shapes_json.append(shape.tolist())
    
    with open('parking_space_config' + str(pointMapHash) + '.json', 'w') as f:
        json.dump(shapes_json, f)
        
def getAverageColorOfParkingSpaces(parking_spaces, image):
    average_colors = []
    i = 1
    # Iterate over each parking space
    for parking_space in parking_spaces:
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [parking_space], 255)
        
        # Get the pixels that make up the rectangle as defined by our configuration file
        masked_img = cv2.bitwise_and(image, image, mask=mask)
                
        average_color = cv2.mean(masked_img, mask=mask)
        
        average_colors.append(average_color)
        
        print("Shape " + str(i) + ": ", average_color)
        i = i + 1
        
    return average_colors
    
def mobileNetSSDObjectDetection(parking_spaces, image):
    occupiedSpaces = []
    parking_space_number = 1
    for parking_space in parking_spaces:
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [parking_space], 255)
        
        # Get the pixels that make up the rectangle as defined by our configuration file
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        
        # Read the pretrained neural network we got from https://github.com/chuanqi305/MobileNet-SSD
        model = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_model.caffemodel')
        
        # Get the top left and bottom right corners of the rectangle
        x1 = parking_space[0][0]
        y1 = parking_space[0][1]
        x2 = parking_space[2][0]
        y2 = parking_space[2][1]
        
        # Crop the image to only the rectangular part defined in our configuration file
        cropped_image = masked_img[y1:y2, x1:x2]
        
        # Put the image into the pretrained model
        blob = cv2.dnn.blobFromImage(cropped_image, scalefactor=1.0, size=(300, 300), mean=(127.5, 127.5, 127.5))
        model.setInput(blob)
        detections = model.forward()
        
        # Iterate over the detections
        space_occupied = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            print(confidence)
            # If there is a legitamate confidence level for an object being detected, mark the space
            if confidence > 0.5:
                space_occupied = True
                # Mark the space in the image
                image = cv2.polylines(image, [parking_space], True, (255, 0, 0), 2)

        if found:
            occupiedSpaces.append(parking_space_number)
            
        parking_space_number = parking_space_number + 1
        
    return occupiedSpaces
        
def torchObjectDetection(parking_spaces, image):
    occupiedSpaces = []
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    #model = torch.hub.load('.', 'custom', path='/media/sf_Shared/car-human.pt', source='local') 
    #model = torch.hub.load('/media/sf_Shared/yolov5-tracking', 'custom', path_or_model='car-human.pt', source='local')
    device = 'cpu'
    model.to(device)
    parking_space_number = 1
    for parking_space in parking_spaces:
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [parking_space], 255)
        
        # Get the pixels that make up the rectangle as defined by our configuration file
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        
        # Get the top left and bottom right corners of the rectangle        
        x1 = parking_space[0][0]
        y1 = parking_space[0][1]
        x2 = parking_space[2][0]
        y2 = parking_space[2][1]
        
        # Crop the image to only the rectangular part defined in our configuration file
        cropped_image = masked_img[y1:y2, x1:x2]
        
        frame = [cropped_image]
        results = model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        classes = model.names
        
        found = False
        n = len(labels)
        # Iterate over the labels
        for i in range(n):
            row = cord[i]
            print(classes[int(labels[i])], row[4])
            # if there is a valid amount of confidence in object detection, mark the parking lot as occupied
            if row[4] >= 0.2:
                bgr = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(image, classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                found = True
        
        if found:
            occupiedSpaces.append(parking_space_number)
        
        parking_space_number = parking_space_number + 1
            
    return occupiedSpaces
                
def postJSONDataToRestful(occupiedSpaces):
    # Define the URL of the RESTful endpoint
    url = 'tbd'

    # Convert the data to a JSON string
    json_data = json.dumps(occupiedSpaces)

    # Set the content type to JSON
    headers = {'Content-Type': 'application/json'}

    # Send the JSON data to the endpoint
    response = requests.post(url, data=json_data, headers=headers)

    # Print the response from the server
    print(response.text)


def highlightParkingSpaces(feed_name, pointMapHash):
    # Load the image
    image = cv2.imread(feed_name)
    
    #plt.imshow(image)
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    
    ### LOADING LOGIC
    
    # Load the vertices from the JSON file
    parking_spaces = loadJSONParkingSpaceData(pointMapHash)
        
    ### END OF LOADING LOGIC
    
    ### CREATING JSON FILES LOGIC
    
    #parking_spaces = [
    #np.array([[233,10],  [396,10],  [396,87],  [233,87]]), 
    #np.array([[233,101], [396,101], [396,177], [233,177]]),]
    
    #createJSONParkingSpaceData(pointMapHash, parking_spaces)
    
    ### END CREATING JSON FILES LOGIC
    
    #mobileNetSSDObjectDetection(parking_spaces, image)
    
    occupiedSpaces = torchObjectDetection(parking_spaces, image)
    
    print(occupiedSpaces)
    
    #postJSONDataToRestful(occupiedSpaces)

    #image = cv2.polylines(image, parking_spaces, True, (255, 0, 0), 2)
    
    # Display the result
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.savefig('result.png')
    plt.show()

@app.route('/run', methods=['GET', 'POST'])
def run():
    highlightParkingSpaces(app.config['arg1'], app.config['arg2']);
    return jsonify({'message': 'Success!'})

if __name__ == '__main__':
    app.run()
