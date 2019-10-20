import qrcode
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


random.seed(4)
qr = qrcode.QRCode(
    version = 1,
    box_size = 15,
    border=5
    )
data = 'http://eazyparking.in/useraccount/Name'
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image(fill = 'black',back_color='white')
img.save("Generated.png")
data = 'http://eazyparking.in/useraccount/Name'
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image(fill = 'black',back_color='white')
img.save("Scanned.png")

original = cv2.imread("Generated.png")
duplicate = cv2.imread("Scanned.png")

# 1) Check if 2 images are equalsaZ
if original.shape == duplicate.shape:
    print("The images have same size and channels")
    difference = cv2.subtract(original, duplicate)
    b, g, r = cv2.split(difference)


    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are completely Equal")

cv2.imshow("Original", original)
cv2.imshow("Duplicate", duplicate)
cv2.waitKey(0)
cv2.destroyAllWindows()