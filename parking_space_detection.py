import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

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
    for parking_space in parking_spaces:
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [parking_space], 255)
        
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        
        average_color = cv2.mean(masked_img, mask=mask)
        
        average_colors.append(average_color)
        
        print("Shape " + str(i) + ": ", average_color)
        i = i + 1
        
    return average_colors
    
def detectObjectsInImage(parking_spaces, image):
    j = 1
    for parking_space in parking_spaces:
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [parking_space], 255)
        
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        
        model = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_model.caffemodel')
        
        x1 = parking_space[0][0]
        y1 = parking_space[0][1]
        x2 = parking_space[2][0]
        y2 = parking_space[2][1]
        
        cropped_image = masked_img[y1:y2, x1:x2]
        
        plt.imshow(cropped_image)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
                
        blob = cv2.dnn.blobFromImage(cropped_image, scalefactor=1.0, size=(300, 300), mean=(127.5, 127.5, 127.5))
        model.setInput(blob)
        detections = model.forward()
        
        space_occupied = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            print(confidence)
            if confidence > 0.5:
                space_occupied = True
                image = cv2.polylines(image, [parking_space], True, (255, 0, 0), 2)

        print("Space Occupied " + str(j) + ": ", space_occupied)
        j = j + 1
        
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
    #np.array([[30,32], [221,32], [221,121], [30,121]]), 
    #np.array([[30,130], [223,130], [223,220], [30,220]])]
    
    #createJSONParkingSpaceData(pointMapHash, parking_spaces)
    
    ### END CREATING JSON FILES LOGIC
    
    detectObjectsInImage(parking_spaces, image)

    #image = cv2.polylines(image, parking_spaces, True, (255, 0, 0), 2)
    
    # Display the result
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.savefig('result.png')
    plt.show()

def main(argv):
    print("Loading image: ", argv[0])
    print("Using configuration: ", argv[1])
    highlightParkingSpaces(argv[0], argv[1]);
    
main(sys.argv[1:])