import cv2

cap = cv2.VideoCapture("echo-hereweare.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )
