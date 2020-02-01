import os
from thresh import *
from tensorflow.keras import datasets, layers, models


# Create a window to display the camera feed
cv2.namedWindow('Camera Output')
cv2.namedWindow('hand')

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)





#os.mkdir(path)
model = models.load_model('machLearnModv2.h5')
symbols = ['Woo', 'Back']

def prepare(imgArr):
    retArr = imgArr.reshape(-1,30,30,1)
    return retArr


def showCamera():

    keyPressed = -1
    takingImg = False
    previous = None
    consecCounter = 0
    while (keyPressed < 0 or keyPressed == 113):  # any key pressed has a value >= 0

        # Grab video frame, decode it and return next video frame
        readSuccess, sourceImage = videoFrame.read()

        sourceImage = cv2.flip(sourceImage, 1)

        cv2.rectangle(sourceImage, (750, 115), (1130, 500), (0, 255, 0), 2)

        hand = sourceImage[115:500, 750:1130]

        hand = preprocess(hand)

        hand = cv2.resize(hand, (30, 30), interpolation=cv2.INTER_AREA)

        cv2.imshow('hand', hand)

        prediction = model.predict(prepare(hand))
        index = np.argmax(prediction)
        if(previous is None):
            previous = index
        elif(index == previous):
            if(consecCounter == 5):
                print(symbols[index])
                consecCounter += 1
            elif(consecCounter < 5):
                consecCounter += 1
        else:
            previous = index
            consecCounter = 0






        # Display the source image
        cv2.imshow('Camera Output', sourceImage)

        # Check for user input to close program
        keyPressed = cv2.waitKey(1)  # wait 1 milisecond in each iteration of while loop

        if (keyPressed == 113):
            takingImg = True
            path = os.getcwd() + "/test_2"
            pictureCount = 1500
        if (takingImg and pictureCount < 3000):
            cv2.imwrite(os.path.join(path, 'img_' + str(pictureCount) + '.jpg'), hand)
            print(pictureCount)
            pictureCount += 1









# Close window and camera after exiting the while loop
showCamera()
cv2.destroyAllWindows()
videoFrame.release()