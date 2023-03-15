import numpy
import os

import pandas as pd
import sklearn
import utils as ut
from sklearn.metrics.cluster import normalized_mutual_info_score, completeness_score
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, homogeneity_score
from sklearn.metrics.cluster import contingency_matrix, adjusted_mutual_info_score,fowlkes_mallows_score
from sklearn.metrics import accuracy_score
import ntpath
import time
from collections import Counter

import warnings
warnings.simplefilter("ignore")

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return numpy.sum(numpy.amax(contingency_matrix, axis=0)) / numpy.sum(contingency_matrix)


def loadResultFile(resFile):
    # print("loading file: ", resFile)
    clusDocMap={}
    clusIDs = []
    docCluster = {}
    with open(resFile) as input:
        line = input.readline()
        while line:
            docIdPredClusId = line.strip().split(' ')  # [documentID, ClusterID]
            currentDocID = docIdPredClusId[0]
            currentCluID = docIdPredClusId[1]
            docCluster[currentDocID] = currentCluID
            if currentCluID not in clusIDs:
                clusIDs.append(currentCluID)
                clusDocMap[currentCluID] = []

            clusterDocs = clusDocMap[currentCluID]
            clusterDocs.append(currentDocID)
            line = input.readline()

        input.close()
    return clusDocMap, clusIDs, docCluster
#End Function

def accuracy(clusDocMap, docsClassMap):
    doc_cluster_map_cluster_to_class_conversion = {}
    for clussID, doccIDs in clusDocMap.items():
        count_instance_of_class = {}
        max_instance = 0
        max_class = ''

        tmp_docsClassMap  = docsClassMap
        # tmp_docsClassMap = {}    # FOR MSTreamF
        # for d_id, c_id in docsClassMap.items():  # FOR MSTreamF
        #     tmp_docsClassMap[str(int(d_id))] = c_id

        for doccID in doccIDs:
            class_name = tmp_docsClassMap[doccID]
            instance_total = count_instance_of_class.get(class_name, 0)
            count_instance_of_class[class_name] = (instance_total + 1)
            if (instance_total + 1) > max_instance:
                max_instance = (instance_total + 1)
                max_class = class_name
        for doccID in doccIDs:
            doc_cluster_map_cluster_to_class_conversion[doccID] = class_name
    return  doc_cluster_map_cluster_to_class_conversion


def evaluate_file(file_path, docsClassMap, classDocMap, stats_dir, file_name, summary_file = None):

    clusDocMap, clusIDs, docCluster = loadResultFile(file_path)
    # Updated for advance evaluation
    originalArray = []
    predictedArray = []

    predictedArrayForAccuracy = []
    doc_class_class_map = accuracy(clusDocMap, docsClassMap)

    for docId, classID in docsClassMap.items():
        # predictedClusterId = docCluster[str(int(docId))]   # FOR MSTreamF
        predictedClusterId = docCluster[docId]
        originalArray.append(classID)
        predictedArray.append(predictedClusterId)

        predictedArrayForAccuracy.append(doc_class_class_map[docId])
        # predictedArrayForAccuracy.append(doc_class_class_map[str(int(docId))])  # FOR MSTreamF

    nmi = normalized_mutual_info_score(originalArray, predictedArray)
    ari = adjusted_rand_score(originalArray, predictedArray)
    v_score = v_measure_score(originalArray, predictedArray)
    com_score = completeness_score(originalArray, predictedArray)
    homo = homogeneity_score(originalArray, predictedArray)
    ami = adjusted_mutual_info_score(originalArray, predictedArray)
    fms = fowlkes_mallows_score(originalArray, predictedArray)
    purity = purity_score(originalArray, predictedArray)
    inverse_purity = purity_score( predictedArray, originalArray)

    acc = accuracy_score(originalArray, predictedArrayForAccuracy)
    # --- updated code enede

    # tool = NMI(classDocMap, clusDocMap, docsClassMap)
    # nmi = tool.calculate()
    dict_ = {}
    dict_['File'] = file_path + "\t "
    dict_['NMI'] =  str(nmi) + " \t "
    dict_['ARI'] =   str(ari) + " \t "
    dict_['V_SCORE'] =   str(v_score) + "\t "
    dict_['COMP'] =   str(com_score) + "\t "
    dict_['Homo'] =   str(homo) + "\t"
    dict_['AMI'] =   str(ami) + " \t "
    dict_['FMS'] =   str(fms) + " \t "
    dict_['Pur'] =   str(purity) + " \t"
    dict_['Acc'] =   str(acc) + "\t"
    dict_['InvPur'] =   str(inverse_purity) + " \t"
    dict_['Clusters'] =   str(clusDocMap.__len__()) + " \t"
    dict_['Classes'] =   str(classDocMap.__len__())

    temp = file_path + "\t " \
           + str(nmi) + " \t "\
           + str(ari) + " \t "\
           + str(v_score) + "\t "\
           + str(com_score) + "\t "\
           + str(homo) + "\t"\
           + str(ami) + " \t "\
           + str(fms) + " \t "\
           + str(purity) + " \t"\
           + str(acc) + "\t"\
           + str(inverse_purity) + " \t"\
           + str(clusDocMap.__len__()) + " \t"\
           + str(classDocMap.__len__())

    console = temp.replace("_ALPHA", "\t").replace("_BETA", "\t").replace(".txt", "")
    print(console)
    if summary_file != None:
        summary_file.write(temp)
        summary_file.write("\n")
    statFile = file_name.replace(".txt", "_STATISTICS.txt")
    f = open(stats_dir + statFile, "w")
    f.write(temp)
    f.write("\n")
    for clusterID, documents in clusDocMap.items():
        temp = str(clusterID) + " \t" + str(documents.__len__()) + " \n"
        f.write(temp)
    f.close()
    pd.DataFrame(dict_).to_csv(statFile.replace(".txt",".csv"))
    return nmi, dict_

def evaluate(classDocMap, docsClassMap, prediction_dir, stats_dir, highest_nmi_by_dir = True):
    ath = stats_dir + "/"
    try:
        os.makedirs(stats_dir)
    except:
        print(stats_dir, " already exists")
    total_files = 0
    header = "File\t NMI \t ARI \tV_Measure \t Completeness\t Homogeneity\t AMI \tFMS \tpurity \t Accuracy \t I_puri \tClusters \t classes \n"
    print(header)
    isFile = os.path.isfile(prediction_dir)
    highest_nmi = 0.0
    highest_nmi_file = ""
    if isFile:
        highest_nmi, _ = evaluate_file(prediction_dir, docsClassMap, classDocMap, stats_dir, ntpath.basename(prediction_dir))
    else:
        for r, directories, list_of_files in os.walk(prediction_dir):  # getting predicted clusters and documents
            summary = r +"/#summary-{0}.tsv".format(str(time.time()))
            summary_file = open(summary,"w")
            summary_file.write(header)
            dir_highest_nmi = 0.0
            dir_highest_nmi_file = ""
            for file in list_of_files:
                if ("#summary" in file):
                    continue
                nmi, _ = evaluate_file(r + "/" + file,docsClassMap, classDocMap, stats_dir, file, summary_file=summary_file)
                total_files+=1
                if nmi > dir_highest_nmi:
                    dir_highest_nmi = nmi
                    dir_highest_nmi_file = (r + "/" + file)
            if highest_nmi_by_dir == True:
                summary_file.write("---------Directory Highest NMI------------- \n")
                summary_file.write(str(dir_highest_nmi_file)+"\t "+str(dir_highest_nmi)+"\n")
                print("---------Directory Highest NMI-------------")
                print(str(dir_highest_nmi_file) + "\t " + str(dir_highest_nmi))
                print("-------------------------------------------")
            if highest_nmi < dir_highest_nmi:
                highest_nmi = dir_highest_nmi
                highest_nmi_file = dir_highest_nmi_file
            summary_file.close()


    print("TOTAL FILES: ",total_files)
    return highest_nmi, highest_nmi_file


def evaluate_results(dataset, prediction_dir, stats_dir):
    # if (not os.path.exists(stats_dir)):
    #     os.makedirs(stats_dir)
    classDocMap, docsClassMap = ut.loadOrigialDocClassLabels(dataset)
    highest_nmi, highest_nmi_file = evaluate(classDocMap,docsClassMap,prediction_dir, stats_dir)
    print("----------- HIGHEST NMI --------")
    print(highest_nmi_file," ", highest_nmi)

if __name__ == '__main__':
    dataDir = "data/"
    # outputPath = "F:/PhD/Coding/OSDM/venv/test/"
    # outputPath = "F:/PhD/Coding/DTM/dtm-master/dtm-master/dtm/News-T-N/model_run/lda-seq/clustering_results.txt"
    # outputPath = "result/MStreamF/News-T-NK0iterNum1SampleNum1alpha0.03beta0.03BatchNum16BatchSaved2.txt"
    statPath = "stats/"


    # dataset = dataDir+"News11104"
    # dataset = dataDir+"News-T11104"
    # dataset = dataDir+"News"
    # dataset = dataDir+"reuters9445"
    # dataset = dataDir+"reuters21578"
    # dataset = dataDir+"reuters21578-T"
    # dataset = dataDir+"Tweets"
    dataset = dataDir+"Tweets-T"
    # dataset = dataDir+"News-T"

    outputPath = "Output_Updated/Tweets-T/_LCB_STC_ICF_WS_CWW_EPB500_EPS30_EPI60/Decay/"
    evaluate_results(dataset,outputPath,statPath)
