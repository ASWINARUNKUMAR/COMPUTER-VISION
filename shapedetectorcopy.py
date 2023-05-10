import cv2
import imutils
from colorlabeler import ColorLabeler

image = cv2.imread(r"C:\Users\Home\Documents\shape detection\shapes.png")
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

(T,thresh) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

print(T)

cv2.imshow("window", thresh)
cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cl = ColorLabeler()
for i,contour in enumerate(cnts):
    if i == 0:
        continue
    epsilon = 0.01*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
    contour = contour.astype("float")
    contour *= ratio
    contour = contour.astype("int")
    
    cv2.drawContours(image, [contour], 0, (255,0,0), 3)
    
    x,y,w,h = cv2.boundingRect(contour)
    x_mid = int(x + w/2)
    y_mid = int(y + h/2)
    
    color = cl.label(lab, contour)
    
    
    if len(approx) == 3:
        print("Triangle")
        cv2.putText(image,color+" triangle", (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    elif len(approx) == 4:
        print("Rectangle")
        cv2.putText(image,color+" rectangle", (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else :
        print("Circle")
        cv2.putText(image,color+" circle", (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imshow("window", image)
    cv2.waitKey(0)

cv2.imshow("window", image)
cv2.waitKey(0)
cv2.destroyAllWindows()