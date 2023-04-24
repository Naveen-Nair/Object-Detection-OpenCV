import cv2

# Load and preprocess image
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

# Detect contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Group contours by object
object_contours = []
for contour in contours:
    # Extract features from contour
    features = cv2.boundingRect(contour)
    # Check if contour belongs to an existing object
    matched = False
    for obj_contour in object_contours:
        obj_features = cv2.boundingRect(obj_contour[0])
        # Check if features match within a threshold
        if abs(features[0]-obj_features[0])<50 and abs(features[1]-obj_features[1])<50:
            obj_contour.append(contour)
            matched = True
            break
    # Create new object if no match found
    if not matched:
        object_contours.append([contour])

# Draw bounding boxes
for obj_contour in object_contours:
    # Compute bounding box of object
    x,y,w,h = cv2.boundingRect(obj_contour[0])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# Display result
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
