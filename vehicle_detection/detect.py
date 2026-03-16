from ultralytics import YOLO
import cv2
model= YOLO("yolo11n.pt") #Load the Model

#Run the model on the image
#results=model("/Users/jay1197/Downloads/retail_shelf_cv/lane_detection/road.jpg")

#Run the model on camera
results=model("/Users/jay1197/Downloads/retail_shelf_cv/lane_detection/solidWhiteRight.mp4",stream=True,conf=0.5)
for result in results:
    frame=result.plot()
    cv2.imshow("detection",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()

# results[0].show()