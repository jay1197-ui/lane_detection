import cv2
import numpy as np
from ultralytics import YOLO

def region_of_interest(edges):
    """Apply a triangular mask to keep only the lane region."""
    height, width = edges.shape
    mask = np.zeros_like(edges)
    triangle = np.array([[[0, height], [width, height], [width // 2, height // 2]]])
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(edges, mask)

def display_lines(image, lines):
    """Draw lines on a blank canvas the same size as the input image."""
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = [int(v) for v in line.reshape(4)]
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
    return line_image


def average_slope_intercept(image, lines):
    """
    Separate Hough lines into left/right lanes by slope sign,
    then average each group into a single representative line.
    """
    left_lines, right_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope, intercept = np.polyfit([x1, x2], [y1, y2], 1)
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    left_avg  = np.average(left_lines,  axis=0) if left_lines  else None
    right_avg = np.average(right_lines, axis=0) if right_lines else None
    return left_avg, right_avg


def make_coordinates(image, line_parameters):
    """Convert slope and intercept into pixel start/end coordinates."""
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def get_lane_overlay(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(blur,50,150)
    cropped=region_of_interest(edges)
    lines=cv2.HoughLinesP(cropped,rho=2,theta=np.pi/180,threshold=100,minLineLength=40,maxLineGap=5)

    if lines is None:
        return np.zeros_like(frame)
    left_avg, right_avg = average_slope_intercept(frame, lines)
    left_line = make_coordinates(frame,left_avg) if left_avg is not None else None
    right_line = make_coordinates(frame,right_avg) if right_avg is not None else None
    averaged_lines = [l for l in [left_line,right_line] if l is not None]
    if not averaged_lines:
        return np.zeros_like(frame)
    return display_lines(frame,np.array(averaged_lines))
if __name__ == "__main__":
    model=YOLO("yolo11n.pt")
    cap=cv2.VideoCapture("media/solidWhiteRight.mp4")
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        results=model(frame,verbose=False)
        yolo_frame=results[0].plot()
        lane_lines=get_lane_overlay(frame)
        combined=cv2.addWeighted(yolo_frame,1,lane_lines,1,0)
        cv2.imshow("ADAS_Line+Vehicle Detection",combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

