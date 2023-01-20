import pickle5 as pk


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