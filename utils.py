import time
import os
import json

def loadOrigialDocClassLabels(dataset, classDocMap={}, docsClassMap={}):
    classIDs = []
    with open(dataset) as input:  # getting original classes and documents , storing docsClassMap, classDocMap
        line = input.readline()
        while line:
            obj = json.loads(line)
            docId = int(obj['Id'])
            docClass = int(obj['clusterNo'])
            if docClass not in classIDs:
                classIDs.append(docClass)
                classDocMap[docClass] = []
            relatedArrayOfDocs = classDocMap[docClass]
            relatedArrayOfDocs.append(docId)
            docsClassMap[obj['Id']] = docClass
            line = input.readline()
    return classDocMap, docsClassMap
#End function

