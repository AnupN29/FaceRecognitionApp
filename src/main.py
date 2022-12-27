from keras_vggface.vggface import VGGFace
from matplotlib import pyplot as plt 
from functions import *

def main(model):
    print("Welcome to FIGURE IT OUT ENTERPRISE")
    while True:
        try:
            com = int(input("Press the following options: \n 1. To scan face to existing members \n 2. To register as new member \n 3. To exit \n"))
        except:
            print("Invalid Input \n")
            continue

        if com == 1:
            featureFileR = open('../SavedFeatures/Features.pkl', 'rb')
            data = pk.load(featureFileR)
            faceRec(model,data)
        elif com == 2:
            newUser(model)
        elif com == 3:
            break
        else:
            print("Invalid Input \n")
 

resnet50Features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg') 


main(resnet50Features)

