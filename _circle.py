import os,cv2,sys,numpy as np
sys.path.insert(0,r"C:\Users\yimingwang218185\Desktop\jieqi_ai")
from yolo_detector import YOLOChessDetector

img=cv2.imread(os.path.expanduser(r"~\Desktop\chess_fuck.png"))
d=YOLOChessDetector()
dets=d.detect(img)
out=os.path.expanduser(r"~\Desktop\test")
os.makedirs(out,exist_ok=True)

for i,dd in enumerate(dets):
    cx,cy=dd["x"],dd["y"]
    x1=max(0,int(cx-dd["w"]/2))
    y1=max(0,int(cy-dd["h"]/2))
    x2=min(img.shape[1],int(cx+dd["w"]/2))
    y2=min(img.shape[0],int(cy+dd["h"]/2))
    full=img[y1:y2,x1:x2]
    if full.size==0:continue
    h,w=full.shape[:2]

    gray=cv2.cvtColor(full,cv2.COLOR_BGR2GRAY)
    blurred=cv2.medianBlur(gray,5)
    circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,dp=1.0,minDist=80,
                              param1=80,param2=30,minRadius=25,maxRadius=55)
    if circles is not None:
        pcx,pcy=map(int,circles[0][0][:2])
    else:
        pcx,pcy=w//2,h//2

    S=int(min(dd["w"],dd["h"])*0.58)
    cx2=max(0,int(x1+pcx-S//2))
    cy2=max(0,int(y1+pcy-S//2))
    cx3=min(img.shape[1],int(x1+pcx+S//2))
    cy3=min(img.shape[0],int(y1+pcy+S//2))
    crop=img[cy2:cy3,cx2:cx3]

    r=int(dd["y"]/img.shape[0]*10)
    c=int(dd["x"]/img.shape[1]*9)
    cv2.imencode(".png",crop)[1].tofile(os.path.join(out,f"{i:02d}_r{r}c{c}.png"))
print(f"Saved {len(dets)} images")
