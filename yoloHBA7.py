from ultralytics import YOLO
import os
import glob
import time
from time import gmtime, strftime
import cv2
import serial
import argparse
import numpy as np
from numpy.linalg import norm

font = cv2.FONT_HERSHEY_SIMPLEX
lastDetectTimer = 0
lastDetectTimer1 = 0
counter = 0
serialMode = 0
hbaStatus = 0
yoloDevice = "cpu"
videoDevice = 2
minConf = 0.2
camWidth = 1920
camHeight = 1080

oldTime = 0
diff = 0

logString = ""
try:
    def brightness(img):
        if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
            return np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(img)
except:
    print("")
def change_brightnes(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Show Video Output", action="store_true")
parser.add_argument("-s", "--save", help="Save the Images", action="store_true")
parser.add_argument("-l", "--log", help="Loging", action="store_true")
parser.add_argument("-c", "--conf", type=float, default=minConf, help="Minimal confidence for detecting Objects")
parser.add_argument("-d", "--device", default=yoloDevice, help="Device for YOLO")
parser.add_argument("-m", "--model", default="/home/jkb/Schreibtisch/runs/detect/train29/weights/best.engine", help="Model for Prediction")
parser.add_argument("-e", "--exposure", default="1", help="Auto Exposure Mode (1 = Manu, 3 = Auto)")
parser.add_argument("-t", "--time", default="700", help="Exposure Time (0-5000)")
parser.add_argument("-b", "--brightness", help="Brightness (-180-180)")
args = parser.parse_args()

if args.log:
    logfile = open('/home/jkb/Schreibtisch/logFile.txt','a')
    logfile.write("---------------------- \n\n")
    logfile.write(str(strftime("%d %b %Y %H:%M:%S", gmtime()) + ": Start..." + "\n"))
   
arguments = str(args)
if args.log:
    logfile.write(str(strftime("%d %b %Y %H:%M:%S", gmtime()) + ": Arguments: " + arguments + "\n"))
print("Arguments: ", arguments)
save = "/home/jkb/Schreibtisch/YOLO/HBA3/rawImagesHBA3/"
log = "/home/jkb/Schreibtisch/"

print("Start...")
if args.verbose:
    print("Verbose")
if args.save:
    print("Save")    
    
try:
    port = "/dev/ttyUSB0"
    serialPort = serial.Serial(port = port, baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
    open(port)
    serialMode = 1
    if args.log:
        logfile.write(str(strftime("%d %b %Y %H:%M:%S", gmtime()) + ": Serial connection established \n"))
except:
    print("Serial connection not possible")
    serialMode = 0
    if args.log:
        logfile.write(str(strftime("%d %b %Y %H:%M:%S", gmtime()) + ": Serial connection not possible \n"))

camera =  cv2.VideoCapture(videoDevice)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
print("Codec: ", camera.get(cv2.CAP_PROP_FOURCC))

exposureTime = args.time
exposureMode = args.exposure
if args.brightness:
    exposureMode = 3

os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=power_line_frequency=1") #Powerline Freq 50hz
os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=auto_exposure=" + str(exposureMode)) #Auto Exposure OFF
os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=exposure_time_absolute=" + str(exposureTime)) #Exposure Time 700
os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=exposure_dynamic_framerate=0") #Dynamic Framerate OFF
if args.brightness:
    os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=brightness=" + str(args.brightness))

if args.log:
        logfile.write(str(strftime("%d %b %Y %H:%M:%S", gmtime()) + ": VideoDevice: "))
        logfile.write(str(videoDevice)) 
        logfile.write(" ")
        logfile.write("Camera Height: ")
        logfile.write(str(camHeight))
        logfile.write(" ")
        logfile.write("Camera Width: ")
        logfile.write(str(camWidth))
        
        logfile.write("\n")
        
time.sleep(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth) #Camera width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight) #Camera height
lastDetectTimer = int(time.time())

# Load a model
#model = YOLO('/home/jkb/Schreibtisch/runs/detect/train29/weights/best.onnx')  # old: /home/jkb/runs/detect/train21/weights/best.pt # OLD OLD
model = YOLO('/home/jkb/Schreibtisch/train2/weights/best.pt')  # old: /home/jkb/runs/detect/train21/weights/best.pt

if args.log:
        logfile.write(str(strftime("%d %b %Y %H:%M:%S", gmtime()) + ": ...DONE \n\n"))
        
while True:
    if args.log:
        logString = str(strftime("%d %b %Y %H:%M:%S", gmtime()) + ": ")
    _, img = camera.read()
    img = cv2.rotate(img, cv2.ROTATE_180)
    if args.save:
        cv2.imwrite(save + "raw/" + str(time.time()).replace(".", "") + ".png", img)
    imgRaw = img.copy()
    img = img[0:850, 0:1920] #Motorhaube Spiegelung Entfernung
    img = img[212:638, 480:1440] #Ausschnitt
    print(img.shape)
    if args.save:
        cv2.imwrite(save + "normal/NOR" + str(time.time()).replace(".", "") + ".png", img)
    imgOri = img.copy()
    
    hue = round(brightness(img), 0)
    print("Hue: ", hue)
    #huePlus = 15 - hue
    #print("huePlus: ", huePlus)
    #img = change_brightnes(img, value=int(35 - hue))
    #hue = round(brightness(img), 0)
    #print("Hue: ", hue)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    results = model(img, device=args.device, conf=args.conf)  # return a list of Results objects

    # Process results list
    for result in results:
        box = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)
    try:
        cords = box.xyxy[0].tolist()
        conf = box.conf.tolist()
        clas = box.cls.tolist()
        leng = cords[2] - cords[0]
        hei = cords[1]
        hig = cords[3] - cords[1]
        print("lenght: ", leng)
        print("height: ", hei)  
        if leng < 500:
            if hei > 120:
                if args.log:
                    logString = logString + " Detect: " + "Class: " + str(clas[0]) + " Conf: " + str(round(conf[0], 2)) + " @ " + str(cords[0]) + "," + str(cords[1]) + "," + str(cords[2]) + "," + str(cords[3]) +" | "
                if lastDetectTimer1 + 1 <= int(time.time()):
                    lastDetectTimer = int(time.time())
        else: 
            if args.log:
                logString = logString + "No relevant Detection" + " | "
        lastDetectTimer1 = int(time.time())
        cords = [round(x) for x in cords]
        print("Cords: ", cords[0])
        print("Conf: ", conf[0])
        print("Class: ", int(clas[0]))
        imageName = str(time.time()).replace(".", "")
        cv2.imwrite("/home/jkb/Schreibtisch/detect/images/" + imageName + ".png", img)
        labelFile = open("/home/jkb/Schreibtisch/detect/labels/" + imageName + ".txt",'w')
        labelFile.write(str(int(clas[0])))
        labelFile.write(" ")
        
        xCenter = cords[0] + (leng / 2)
        xCenter = xCenter / 640
        yCenter = cords[1] + (hig / 2)
        yCenter = yCenter / 640
        print("xCenter: ", xCenter)
        print("yCenter: ", yCenter)
        xWidth = leng / 640
        yHeight = hig / 640
        
        
        labelFile.write(str(xCenter))
        labelFile.write(" ")
        labelFile.write(str(yCenter))
        labelFile.write(" ")
        labelFile.write(str(xWidth))
        labelFile.write(" ")
        labelFile.write(str(yHeight))
        labelFile.close()
        
        #classPredictString = ",".join(str(element) for element in clas)
        #confPredictString = ",".join(str(element) for element in conf)
        if args.verbose:
            img = cv2.rectangle(img, (cords[0] - 13, cords[1] - 13), (cords[2] + 13, cords[3] + 13 ), (0, 255, 255), 3)#
            #img = cv2.putText(img, preString1, (cords[2] + 13, cords[3] + 13), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
    except:
        print("No Detect")
        if args.log:
            logString = logString + "No Detection" + " | "
    if args.verbose:
        #img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow("Result", img)
        cv2.imshow("RAW", imgRaw)
    print("Last Detect: ", lastDetectTimer)
    print("Time: ", int(time.time()))

    if lastDetectTimer + 3 <= int(time.time()):
        print("HBA Status: 1")
        hbaStatus = 1
        if args.log:
            logString = logString + "HBA Status: " + str(hbaStatus) + " | "
        if serialMode == 1: 
            try:
                serialPort.write(b"1\n")
            except:
                serialMode = 0
                print("Serial not Possible!")    
    else:
        print("HBA Status: 0")
        hbaStatus = 0
        if args.log:
            logString = logString + "HBA Status: " + str(hbaStatus) + " | "
        if serialMode == 1:
            try:
                serialPort.write(b"0\n")
            except:
                serialMode = 0
                print("Serial not Possible!")
        if args.save:        
            cv2.imwrite(save + "normal/OFF/OFF" + str(time.time()).replace(".", "") + ".png", imgOri)

    k = cv2.waitKey(1)
    if k == ord('c'):
        break
    print("Now: ", time.time_ns())
    diff = time.time_ns() - oldTime
    print("DIFF: ", round(diff / 1000000000, 4))
    oldTime = time.time_ns()
    if args.log: 
        print("Log: ", logString)
        logString = logString + " DIFF: " + str(round(diff / 1000000000, 4))
        logfile.write(logString + "\n")

cv2.destroyAllWindows()
if args.log:
    logfile.write("-END- \n")
    logfile.close()
