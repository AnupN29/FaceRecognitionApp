from mtcnn.mtcnn import MTCNN

import cv2
from scipy.spatial import distance




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






