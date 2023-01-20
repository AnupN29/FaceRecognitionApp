from keras_vggface.vggface import VGGFace
import pickle5 as pk

from matplotlib import pyplot as plt 

from utils import saveFeatures,faceVerify, newUser

def main(model):
    print("\n \n \n Welcome to FIGURE IT OUT ENTERPRISE")
    while True:
        try:
            com = int(input("Press the following options: \n 1. To verify as existing member \n 2. To register as new member \n 3. To exit \n"))
        except:
            print("Invalid Input \n")
            continue

        if com == 1:
            featureFileR = open('SavedFeatures/Features.pkl', 'rb')
            savedFeatures = pk.load(featureFileR)
            name = input("Enter name \n")
            flag =0
            for user in savedFeatures:
                if user['name'] == name:
                    flag =1
                    savedFeatures = user['features']
            if flag == 1:    
                faceVerify(model,savedFeatures,name)
            else:
                print("No such name exist try again \n")
        elif com == 2:
            name = input("Enter name \n")
            newUser(model,name)
        elif com == 3:
            break
        else:
            print("Invalid Input \n")
 

resnet50Features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg') 


main(resnet50Features)

