import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import pickle5 as pk
from scipy.spatial import distance
from datetime import datetime

def extractFace(frame):
    detector = MTCNN(min_face_size = 120)
    faces = []
    results = detector.detect_faces(frame)
    if len(results) > 0:
        for face in results:
            x, y, width, height = face['box']
            face = frame[y:y+height, x:x + width]
            faces.append(face)
    return faces

def extractFeatures(face,model):
    face = cv2.resize(face,(224,224))
    face = face/255.0
    face = face.reshape(1,224,224,3)
    features =model.predict(face)
    return features

def calDist(calFeatures,saveFeatures):
    Features1 = calFeatures.reshape(2048,)
    dist = []
    for features in saveFeatures:
        Features2 = features.reshape(2048,)
        dist.append(distance.euclidean(Features1, Features2)) 
    return min(dist)

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
        frame = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Verify Face', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()






def saveFeatures(features,name):
    flag = 1
    featureFileR = open('../SavedFeatures/Features.pkl', 'rb')
    data = pk.load(featureFileR)
    for user in data:
        if name == user['name']:
            user['features'] = features
            flag = 0
            break
    if flag:
        data.append({"name": name, "features": features})
    featureFileW = open('../SavedFeatures/Features.pkl', 'wb')
    pk.dump(data,featureFileW)




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




