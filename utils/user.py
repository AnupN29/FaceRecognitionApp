import cv2

from datetime import datetime

from .extract import  extractFace, extractFeatures, calDist
from .save_features import saveFeatures

def faceVerify(model,savedFeatures,name):
    cap = cv2.VideoCapture(0)
    t1 = datetime.now()
    text = '''Scaning Your Face'''
    while cap.isOpened():
        ret, frame = cap.read()
        t2 = datetime.now()
        x = t2 - t1         
        if x.seconds==5:
            if type(frame) != type(None):            
                facesExt = extractFace(frame)
                if len(facesExt)==1:
                    featuresExt = extractFeatures(facesExt[0],model)
                    dist = calDist(featuresExt,savedFeatures)
                    if dist < 0.08:
                        print("Verified \n Welcome "+name)
                        break
                    else:
                        t1 = datetime.now()
                        text = '''Trying Again'''
                        continue
                else:
                    t1 = datetime.now()
                    text = '''Trying Again'''
                    continue
                    
        frame = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Verify Face', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def newUser(model,name):
    cap = cv2.VideoCapture(0)
    text = '''Press "a" to capture face'''
    featuresExtAll = []
    while cap.isOpened():
        ret, frame = cap.read()         
        if cv2.waitKey(1) & 0XFF == ord('a'):
            if type(frame) != type(None):            
                facesExt = extractFace(frame)
                if len(facesExt)==1:
                    featuresExt = extractFeatures(facesExt[0],model)
                    featuresExtAll.append(featuresExt)
                    text = '''Press "a" to capture different pose '''
        frame = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Add New User', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    saveFeatures(featuresExtAll,name)
    print("Successfully Added new member \n")
