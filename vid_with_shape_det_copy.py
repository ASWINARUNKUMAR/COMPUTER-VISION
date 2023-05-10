import cv2
import imutils
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class ColorLabeler:
    def __init__(self):
   		# initialize the colors dictionary, containing the color
   		# name as the key and the RGB tuple as the value
   		colors = OrderedDict({
   			"red": (255, 0, 0),
   			"green": (0, 255, 0),
   			"blue": (0, 0, 255)})
   		# allocate memory for the L*a*b* image, then initialize
   		# the color names list
   		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
   		self.colorNames = []
   		# loop over the colors dictionary
   		for (i, (name, rgb)) in enumerate(colors.items()):
   			# update the L*a*b* array and the color names list
   			self.lab[i] = rgb
   			self.colorNames.append(name)
   		# convert the L*a*b* array from the RGB color space
   		# to L*a*b*
   		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
        
    def label(self, image, c):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)
            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                minDist = (d, i)
        # return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]
  
  
# define a video capture object
vid = cv2.VideoCapture(r"C:\Users\Home\Downloads\test1.mp4")

# Check if camera opened successfully
if (vid.isOpened()== False): 
  print("Error opening video stream or file")
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cl = ColorLabeler()
  
    if ret == True:

        resized = imutils.resize(frame, width=300)
        ratio = frame.shape[0] / float(resized.shape[0])

        gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

        (T,thresh) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(cnts):
            if i == 0:
                continue
            epsilon = 0.01*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # multiply the contour (x, y)-coordinates by the resize ratio,
        	# then draw the contours and the name of the shape on the image
            contour = contour.astype("float")
            contour *= ratio
            contour = contour.astype("int")
            
            cv2.drawContours(frame, [contour], -1, (255,0,0), 2)
            
            x,y,w,h = cv2.boundingRect(contour)
            x_mid = int(x + w/2)
            y_mid = int(y + h/2)
            color = cl.label(lab, contour)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 4)
            
            
            if len(approx) == 3:
                print("Triangle")
                cv2.putText(frame, color+" triangle", (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif len(approx) == 4:
                print("Rectangle")
                cv2.putText(frame, color+" rectangle", (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else :
                print("Circle")
                cv2.putText(frame, color+" circle", (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the resulting frame
        cv2.imshow('Frame',frame)
     
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
     
      # Break the loop
    else: 
        break
    
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()