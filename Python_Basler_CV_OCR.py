import cv2
import pytesseract
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer
from pytesseract import Output
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

'''# Pypylon get camera by serial number
serial_number = '23610391'
info = None
for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    if i.GetSerialNumber() == serial_number:
        info = i
        print('Camera found')
        break
else:
    print('Camera with {} serial number not found'.format(serial_number))

# VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
if info is not None:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
    camera.Open()

'''
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

img = cv2.imread(r'Picture5.jpg')



y1=0
x1=0
h1=35
w1=350
crop_img1 = img[y1:y1+h1, x1:x1+w1]
#cv2.imshow('crop_img1', crop_img1)

y2=65
x2=305
h2=30
w2=170
crop_img2 = img[y2:y2+h2, x2:x2+w2]
#cv2.imshow('crop_img2', crop_img2)


y3=135
x3=302
h3=30
w3=150
crop_img3 = img[y3:y3+h3, x3:x3+w3]
#cv2.imshow('crop_img3', crop_img3)


y4=105
x4=60
h4=60
w4=120
crop_img4 = img[y4:y4+h4, x4:x4+w4]
#cv2.imshow('crop_img4', crop_img4)


'''y5=y4+20
x5=60
h5=30
w5=120
crop_img5 = img[y5:y5+h5, x5:x5+w5]
#cv2.imshow('crop_img5', crop_img5)'''


y6=190
x6=40
h6=80
w6=200
crop_img6 = img[y6:y6+h6, x6:x6+w6]
#cv2.imshow('crop_img6', crop_img6)


'''y7=y6+15
x7=40
h7=20
w7=160
crop_img7 = img[y7:y7+h7, x7:x7+w7]
#cv2.imshow('crop_img7', crop_img7)


y8=y7 +15
x8=40
h8=20
w8=200
crop_img8 = img[y8:y8+h8, x8:x8+w8]
#cv2.imshow('crop_img8', crop_img8)'''


y9=130
x9=190
h9=30
w9=90
crop_img9 = img[y9:y9+h9, x9:x9+w9]
#cv2.imshow('crop_img9', crop_img9)

y10=300
x10=350
h10=30
w10=130
crop_img10 = img[y10:y10+h10, x10:x10+w10]
#cv2.imshow('crop_img9', crop_img9)


#cv2.imshow('crop_img9', crop_img9)

gray = get_grayscale(img)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img1, config=custom_config)

print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img2, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img3, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img4, config=custom_config)
print('Read result= ',text)

'''# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img5, config=custom_config)
print('Read result= ',text)'''

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img6, config=custom_config)
print('Read result= ',text)

'''# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img7, config=custom_config)
print('Read result= ',text)'''

'''# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img8, config=custom_config)
print('Read result= ',text)'''

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img9, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img10, config=custom_config)
print('Read result= ',text)

img = cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 2)

img = cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
img = cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), (0, 255, 0), 2)
img = cv2.rectangle(img, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 0), 2)
'''img = cv2.rectangle(img, (x5, y5), (x5 + w5, y5 + h5), (0, 255, 0), 2)'''
img = cv2.rectangle(img, (x6, y6), (x6 + w6, y6 + h6), (0, 255, 0), 2)
'''img = cv2.rectangle(img, (x7, y7), (x7 + w7, y7 + h7), (0, 255, 0), 2)
img = cv2.rectangle(img, (x8, y8), (x8 + w8, y8 + h8), (0, 255, 0), 2)'''
img = cv2.rectangle(img, (x9, y9), (x9 + w9, y9 + h9), (0, 255, 0), 2)
img = cv2.rectangle(img, (x10, y10), (x10 + w10, y10 + h10), (0, 255, 0), 2)
#cv2.imshow('img', img)

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img10, config=custom_config)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img3 = image.GetArray()
        #gray = get_grayscale(img3)
        #img3 = thresholding(gray)
        y11 = 340
        x11 = 100
        h11 = 65
        w11 = 600
        crop_img11 = img3[y11:y11 + h11, x11:x11 + w11]
        org1 = (x11, y11-20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 255, 255)
        thickness = 2
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(crop_img11, config=custom_config)
        img3 = cv2.putText(img3, text, org1, font, fontScale, color, thickness, cv2.LINE_AA)
        img3 = cv2.rectangle( img3, (x11, y11), (x11 + w11, y11 + h11), (255, 255, 0), 2)

        y12 = 470
        x12 = 670
        h12 = 50
        w12 = 330
        crop_img12 = img3[y12:y12 + h12, x12:x12 + w12]
        org2 = (x12-200, y12-20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 255, 255)
        thickness = 2
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(crop_img12, config=custom_config)
        img3 = cv2.putText(img3, text, org2, font, fontScale, color, thickness, cv2.LINE_AA)
        img3 = cv2.rectangle( img3, (x12, y12), (x12 + w12, y12 + h12), (255, 255, 0), 2)

        y13 = 700
        x13 = 200
        h13 = 150
        w13 = 400
        crop_img13 = img3[y13:y13 + h13, x13:x13 + w13]
        org3 = (x13-100, y13-20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (0, 255, 255)
        thickness = 2
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(crop_img13, config=custom_config)
        img3 = cv2.putText(img3, text, org3, font, fontScale, color, thickness, cv2.LINE_AA)
        img3 = cv2.rectangle( img3, (x13, y13), (x13 + w13, y13 + h13), (255, 255, 0), 2)

        y14 = 880
        x14 = 800
        h14 = 70
        w14 = 200
        crop_img14 = img3[y14:y14 + h14, x14:x14 + w14]
        org4 = (x14-200, y14-20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 255, 255)
        thickness = 2
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(crop_img14, config=custom_config)
        img3 = cv2.putText(img3, text, org4, font, fontScale, color, thickness, cv2.LINE_AA)
        img3 = cv2.rectangle( img3, (x14, y14), (x14 + w14, y14 + h14), (255, 255, 0), 2)

        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img3)


        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()


