import cv2
import re
import datetime
from geopy.geocoders import Nominatim
from spellchecker import SpellChecker
from PIL import Image, ImageEnhance
import numpy as np
import pytesseract
from scipy import ndimage, misc
from PIL import Image
from PIL import ImageEnhance, ImageFilter
import image
import pandas as pd
from pandas import ExcelWriter
from openpyxl import load_workbook
from geopy.extra.rate_limiter import RateLimiter

#Functions
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR4.0/tesseract.exe'
path = "C:/Users/Tayab/PycharmProjects/MediaProject/Kasten_2_Klasse_2-01.2_0212.png"

# Reading Image
img = cv2.imread(path)

#Image Resizing
OCR = image_resize(img, height=650)

#ImageCropping
gray = cv2.cvtColor(OCR, cv2.COLOR_BGR2GRAY) # convert to grayscale
# threshold to get just the signature
retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
# find where the signature is and make a cropped region
points = np.argwhere(thresh_gray == 0) # find where the black pixels are
points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
x, y, w, h = x+25, y+25, w-60, h-60 # make the box a little bigger
crop = gray[y:y+h, x:x+w] # create a cropped region of the gray image
cv2.imshow('imgCrop', crop)

#ImagePreProcessing

thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# Remove horizontal lines
result = crop.copy()
kernel = np.ones((1,3), np.uint8)  # note this is a horizontal kernel
d_im = cv2.dilate(thresh, kernel, iterations=1)
e_im = cv2.erode(d_im, kernel, iterations=1)
close = cv2.morphologyEx(e_im, cv2.MORPH_CLOSE, kernel, iterations=1)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
remove_horizontal = cv2.morphologyEx(e_im, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(result, [c], -1, (255,255,255), 1)

ret,thresh2 = cv2.threshold(result,180,255,cv2.THRESH_BINARY)
thresh2 = thresh2


### Tesserat
custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6'
custom_oem_psm_config = "--oem 3 --psm 3"
text = pytesseract.image_to_string(thresh2, config=custom_oem_psm_config, lang='deu')

## RegEx
res = text.translate({ord(i): None for i in '$°+'})
res.replace(" ", "")
res = re.sub(r"\s+", "", res)
print(res)
date = re.findall("([0-9]{2}\.[0-9]{1}\.[0-9]{4})", str(res))
classification = re.findall("([0-9]{1}\-[0-9]{2}\,[0-9]{1})", str(res))
city = re.findall("Drosden", str(res))
print(date)
print(classification)

cv2.waitKey(0)
cv2.destroyAllWindows()