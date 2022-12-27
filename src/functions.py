import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import pickle5 as pk
from scipy.spatial import distance
 
  
def extractFace(frames):
    detector = MTCNN()
    faces = []
    for i in range(len(frames)):
        temp=[]
        for j in range(len(frames[i])):
            frame =frames[i][j]
            results = detector.detect_faces(frame)
            if len(results) > 0:
                for face in results:
                    x, y, width, height = face['box']
                    face = frame[y:y+height, x:x + width]
                    try:
                        temp.append(face)
                    except:
                        pass
        faces.append(temp)
    return faces

def extractFeatures(faces,model):
    features = []
    numFaces = len(faces)
    for i in range(len(faces)):
        sumFaces = np.zeros(2048)
        for j in range(len(faces[i])):
            face = faces[i][j]
            face = cv2.resize(face,(224,224))
            face = face.reshape(1,224,224,3)
            sumFaces = sumFaces + model.predict(face)
        x = float(len(faces[i]))
        features.append(sumFaces/x)
    return features


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

def calDist(calFeatures,preFeatures):
    temp =[]
    for i in range(len(calFeatures)):
        Features1 = calFeatures[i].reshape(2048,)
        for j in range(len(preFeatures)):
            Features2 = preFeatures[j].reshape(2048,)
            temp.append(distance.euclidean(Features1, Features2)) 
    return np.min(temp)


def newUser(model):
    name = input("Enter your name \n")
    cap = cv2.VideoCapture(0)
    frames = []
    text = '''Press "a" to capture face'''
    while cap.isOpened():
        ret, frame = cap.read()         
        if cv2.waitKey(1) & 0XFF == ord('a'):
            temp =[]
            i = 0
            while i <10:
                ret, frame = cap.read()         
                try:
                    temp.append(frame)
                    i = i +1
                except:
                    pass
            frames.append(temp)
            text = '''Press "a" to capture different pose '''
        frame = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Capture Face', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    faces= extractFace(frames)
    features = extractFeatures(faces,model)
    ok =1
    for f in features:
        if np.isnan(f).any():
            ok = 0
    if ok == 0:
        print("Try again \n")
        newUser(model)
    
        
    saveFeatures(features,name)
    print("Successfully Added new member \n")


def faceRec(model,data):
    cap = cv2.VideoCapture(0)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()         
        if cv2.waitKey(1) & 0XFF == ord('s'):
            temp=[]    
            temp.append(frame)   
            frames.append(temp)
            break    
        frame = cv2.putText(frame, '''Press "s" to scan face''', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Scan Face', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    faces= extractFace(frames)
    features = extractFeatures(faces,model)
    temp=[]
    for i in range(len(data)):
        preFeatures = data[i]["features"]
        try:
            x = calDist(features,preFeatures)
            temp.append(x)
        except:
            faceRec(model,data)
    try:        
        err = np.min(temp)
    
        if err<71:
            user = temp.index(err)
            user = data[user]['name']
            print("Welcome " + user + "\n")
            print(err)
        else:
            print("Sorry member not found \n")
            print(err)
    except:
        pass


        
    