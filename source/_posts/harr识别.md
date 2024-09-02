---
title: harr识别
date: 2024-05-27 15:18:31
tags:
categories:
    - opencv
---

harr识别

<!--more-->

- Haar人脸识别方法



```python
import cv2
import numpy as np
facer = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

img = cv2.imread('./image/face2.JPG')
print(img.shape)

scale_percent = 20 # 图像缩小到原来的50%
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
gray=cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)

facer = facer.detectMultiScale(gray,1.1,5)
i=0
for(x,y,w,h) in facer:
    cv2.rectangle(resized_img,(x,y),(x+w,y+h),(0,0,255),2)
    roi=gray[y:y+h,x:x+w]
    eyes=eye.detectMultiScale(roi,1.1,5)

    for (xeye,yeye,weye,heye) in eyes:
        cv2.rectangle(resized_img,(x+xeye,y+yeye),(x+xeye+weye,y+yeye+heye),(255,0,0),2)

    # i=i+1
    # winname='face'+str(i)
    # cv2.imshow(winname,roi)
cv2.imshow('img',resized_img)
cv2.waitKey()
cv2.destroyAllWindows()

```

![QQ20240527152442.png](https://s2.loli.net/2024/05/27/HsRaKbD4GWwzULM.png)