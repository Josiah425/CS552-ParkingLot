from flask import Flask, jsonify, request
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
import sys
import torch
import boto3
import requests

app = Flask(__name__)

plt.rcParams["figure.figsize"] = [13.91, 9.51]
plt.rcParams["figure.autolayout"] = True

rekognition = boto3.client('rekognition')
s3 = boto3.client('s3')

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
    
def detectCarsWithAWSRekognition(feed_name):
    with open(feed_name, 'rb') as image:
        image_bytes = image.read()

    response = rekognition.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=10, MinConfidence=70)
    
    print(response)
    
    print('Labels detected:')
    instances = [];
    for label in response['Labels']:
        print(f"{label['Name']} : {label['Confidence']}%")
        if label['Name'] == 'Car':
            instances = label['Instances']
        
    cars = [label for label in response['Labels'] if label['Name'] == 'Car']
    
    print(f"Number of cars detected: {len(instances)}")
    
    return len(instances)
    
def postDataToRestfulEndpoint(feed_name, total_parking_spaces, empty_parking_spaces):
    url = "https://d75i8bz9h1.execute-api.us-west-1.amazonaws.com/prod"
    character = feed_name.split('ParkingLot')[1][0]
    
    # Jsonify the data in the format restful endpoint expects
    data = {
        "lotId":character,
        "total":total_parking_spaces,
        "freeSpace":empty_parking_spaces
    }
    
    # Convert the parking lot data to a JSON string
    json_data = json.dumps(data)
    
    # Let the requests know the type of data being sent
    headers = {'Content-Type':'application/json'}
    
    # Send the POST request
    response = requests.post(url, data=json_data, headers=headers)
    
    print(response.status_code)
    print(response.content)
    

def highlightParkingSpaces(feed_name, pointMapHash, image): 
    # Load the vertices from the JSON file
    parking_spaces = loadJSONParkingSpaceData(pointMapHash)
    
    # Call AWS Rekognition to detect cars in the image
    cars_detected = detectCarsWithAWSRekognition(feed_name)
    
    # Send data to a restful endpoint that describes the lot id, total spaces, and the number of empty spaces
    postDataToRestfulEndpoint(feed_name, len(parking_spaces), max(0, len(parking_spaces)-cars_detected))
        
    # Display the result
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.savefig('result.png')
    plt.show()

@app.route('/run', methods=['GET', 'POST'])
def run():
    # Example Url use to call the flask application
    #    127.0.0.1:5000/run?path_to_image=ParkingLotA.jpg&config_number_to_use=1
    #    127.0.0.1:5000/run?path_to_image=ParkingLotB.jpg&config_number_to_use=2
    #    127.0.0.1:5000/run?path_to_image=ParkingLotC.jpg&config_number_to_use=3
    # The path_to_image and config_number_to_use are used here to determine the arguments to the script
    path_to_image = request.args.get('path_to_image')
    config_number_to_use = request.args.get('config_number_to_use')


    response = s3.get_object(
        Bucket='parking-lot-images-cs-final-project-552',
        Key=path_to_image
    )

    contents = response['Body'].read()
    
    # Create the image by converting the data from S3 into a format opencv understands
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    highlightParkingSpaces(path_to_image, config_number_to_use, img);

    # highlightParkingSpaces(path_to_image, config_number_to_use);
    return jsonify({'message': 'Success!'})

if __name__ == '__main__':
    app.run()
