import os
import signal
import glob
import logging
import time
from time import gmtime, strftime
import cv2
import serial
import sys
import argparse
import numpy as npk
from numpy.linalg import norm
from ultralytics import YOLO
from ultralytics import settings

#TEST AUS VSCODE
settings.update({'sync': False})
startTime = str(time.time()).replace(".", "")

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

def log_exception(exc_type, exc_value, exc_traceback):
    print("Unhandled exception:", exc_type, exc_value)
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    end()

sys.excepthook = log_exception

def keyboard_interrupt_handler(signal, frame):
    logger.info("Manual end CTR-C")
    end()

def end():
    print("---END---")
    cv2.destroyAllWindows()
    logger.critical("---END---\n")
    sys.exit(0)

signal.signal(signal.SIGINT, keyboard_interrupt_handler)

logging.basicConfig(filename=str("/home/jkb/Schreibtisch/HBA/v8/logs/" + startTime + '.log'), level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HBA')

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="Show Video Output", action="store_true")
parser.add_argument("-s", "--save", help="Save the Caputerd Images", action="store_true")
parser.add_argument("-c", "--conf", type=float, default=minConf, help="Minimal confidence for detecting Objects")
parser.add_argument("-d", "--device", default=yoloDevice, help="Device for YOLO")
parser.add_argument("-m", "--model", default="/home/jkb/Schreibtisch/train2/weights/best.pt", help="Model for Prediction") #/home/jkb/Schreibtisch/runs/detect/train29/weights/best.engine
parser.add_argument("-e", "--exposure", default="1", help="Auto Exposure Mode (1 = Manu, 3 = Auto)")
parser.add_argument("-t", "--time", default="700", help="Exposure Time (0-5000)")
parser.add_argument("-b", "--brightness", help="Brightness (-180-180)")
parser.add_argument("-p", "--path",default="/home/jkb/Schreibtisch/HBA/v8/", help="Path for Saving")
args = parser.parse_args()

logger.critical('---START---')
logger.info('Start Time: {}'.format(str(startTime)))
logger.info('Arguments: {}'.format(str(args)))

print("Start...")  
print("Arguments: ", str(args))

def configure_serial_connection():
    global serialMode, serialPort
    try:
        port = "/dev/ttyUSB0"
        serialPort = serial.Serial(port = port, baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
        open(port)
        serialMode = 1
        return True
    except:
        print("Serial connection not possible")
        serialMode = 0
        return False

if configure_serial_connection() == True:
    logger.info("configure_serial_connection: OK")
else:
    logger.error("configure_serial_connection: ERROR!")

def initialize_camera():
    global camera
    try:
        camera =  cv2.VideoCapture(videoDevice)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        exposureTime = args.time
        exposureMode = args.exposure
        if args.brightness:
            exposureMode = 3

        if os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=power_line_frequency=1") == 0: #Powerline Freq 50hz
            None
        else:
            return False
        os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=auto_exposure=" + str(exposureMode)) #Auto Exposure OFF
        os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=exposure_time_absolute=" + str(exposureTime)) #Exposure Time 700
        os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=exposure_dynamic_framerate=0") #Dynamic Framerate OFF
        if args.brightness:
            os.system("v4l2-ctl --device /dev/video" + str(videoDevice) + " --set-ctrl=brightness=" + str(args.brightness))
                
        time.sleep(1)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth) #Camera width
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight) #Camera height
        return True
    except:
        return False

if initialize_camera() == True:
    logger.info('initialize_camera: OK')
else:
    logger.error('initialize_camera: ERROR!')

lastDetectTimer = int(time.time())

model = YOLO(args.model) 

logger.info("---initialisierung abgeschlossen--- \n")

def get_active_area(cords):
    logger.info('cord0: {} cord1: {} cord2: {} cord3: {}'.format(str(cords[0]), str(cords[1]), str(cords[2]), str(cords[3]),))
    leng = cords[2] - cords[0]
    hei = cords[1]
    hig = cords[3] - cords[1]
    print("lenght: ", leng)
    print("height: ", hei)  
    logger.info('Len: {} Hei: {}'.format(str(leng), str(hei)))
    if leng < 500:
        if hei > 160:
            return True
    else: 
        return False

def make_label(nameTime, clas):
    labelFile = open(args.path + "labels/detections/" + nameTime + ".txt",'w')
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
    print(str(xCenter), " ", str(yCenter), " ", str(xWidth), " ", str(yHeight))
    labelFile = open(args.path + "labels/detections/" + nameTime + ".txt",'a')
    labelFile.write(str(str(xCenter) + " " + str(yCenter) + " " + str(xWidth) + " " + str(yHeight)))
    labelFile.close()

while True:
    logger.info("-Frame START-")
    nameTime = str(time.time()).replace(".", "")
    _, img = camera.read()
    img = cv2.rotate(img, cv2.ROTATE_180)
    if args.save:
        cv2.imwrite(args.path + "images/raw/" + nameTime + ".png", img)
    imgRaw = img.copy()
    img = img[0:850, 0:1920] #Motorhaube Spiegelung Entfernung
    img = img[212:638, 480:1440] #Ausschnitt
    print(img.shape)
    if args.save:
        cv2.imwrite(args.path + "images/crop/" + nameTime + ".png", img)
    imgOri = img.copy()
    
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    results = model(img, device=args.device, conf=args.conf)  # return a list of Results objects

    # Process results list
    for result in results:
        box = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
    print("Detections: ", len(box))
    logger.info("Detections: {}".format(len(box)))
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)
    if len(box) > 0:
        cords = box.xyxy[0].tolist()
        conf = box.conf.tolist()
        clas = box.cls.tolist()
        leng = cords[2] - cords[0]
        hei = cords[1]
        hig = cords[3] - cords[1]
        print("lenght: ", leng)
        print("height: ", hei)  
        for i in range(len(box)):
            in_active_area = get_active_area(box.xyxy[i].tolist())
            print("In active area " + str(i) + ":" + str(in_active_area))
            logger.info('In active area {} : {}'.format(str(i), in_active_area))
        if leng < 500:
            if hei > 160:
                #Sinvolle Detektion 
                if lastDetectTimer1 + 1 <= int(time.time()):
                    lastDetectTimer = int(time.time())
        else: 
            logger.info("No usefull detection")
            None
        
        lastDetectTimer1 = int(time.time())
        cords = [round(x) for x in cords]
        print("Cords: ", cords[0])
        print("Conf: ", conf[0])
        print("Class: ", int(clas[0]))
        cv2.imwrite(args.path + "images/detections/" + nameTime + ".png", img)
        make_label(nameTime, cords)
        if args.video:
            for i in range(len(box)):
                cords = box.xyxy[i].tolist()
                cords = [round(x) for x in cords]
                print("Cords: ", cords[0])
                if get_active_area(cords) == None:
                    img = cv2.rectangle(img, (cords[0] - 13, cords[1] - 13), (cords[2] + 13, cords[3] + 13 ), (0, 255, 255), 3)#
                    img = cv2.putText(img, str(clas[i]).replace(".0", ""), (cords[2] + 20, cords[3] + 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                elif int(clas[i]) == 0:
                    img = cv2.rectangle(img, (cords[0] - 13, cords[1] - 13), (cords[2] + 13, cords[3] + 13 ), (0, 0, 255), 3)#
                    img = cv2.putText(img, str(clas[i]).replace(".0", ""), (cords[2] + 20, cords[3] + 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    img = cv2.rectangle(img, (cords[0] - 13, cords[1] - 13), (cords[2] + 13, cords[3] + 13 ), (255, 255, 255), 3)#
                    img = cv2.putText(img, str(clas[i]).replace(".0", ""), (cords[2] + 20, cords[3] + 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        print("No Detect")

    if args.video:
        #img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow("Result", img)
        cv2.imshow("RAW", imgRaw)
    print("Last Detect: ", lastDetectTimer)
    print("Time: ", int(time.time()))

    if lastDetectTimer + 3 <= int(time.time()):
        print("HBA Status: 1")
        hbaStatus = 1
        logger.info("HBA Status: 1")
        if serialMode == 1: 
            try:
                serialPort.write(b"1\n")
            except:
                serialMode = 0
                print("Serial not Possible!")    
    else:
        print("HBA Status: 0")
        hbaStatus = 0
        logger.info("HBA Status: 0")
        if serialMode == 1:
            try:
                serialPort.write(b"0\n")
            except:
                serialMode = 0
                print("Serial not Possible!")
        if args.save:        
            cv2.imwrite(args.path + "images/off/" + nameTime + ".png", imgOri)

    k = cv2.waitKey(1)
    if k == ord('c'):
        logger.info("Manual end C in CV2")
        end()
    print("Now: ", time.time_ns())
    diff = time.time_ns() - oldTime
    print("DIFF: ", round(diff / 1000000000, 4))
    oldTime = time.time_ns()
    logger.info("DIFF: " + str(round(diff / 1000000000, 4)))
    logger.info("-Frame END-\n")