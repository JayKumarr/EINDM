from Document import Document
from Model import Model
import numpy as np
import json, time, sys, os, ntpath, argparse
from PrintingFile import print_dic
import psutil
from evaluate import evaluate_results
from random import sample


def printExecutionTime(startTime, str="", divide_by=1):
    print(str+ " time elapsed: {:.2f}s".format( (time.time() - startTime)/divide_by ))
    return time.time()

def output_directory_path(resultDir, dataset, outputPrefix, decay, create_dir=False, merge_old_cluster =False):
    path = resultDir + dataset + "/" + outputPrefix + "/"
    if (decay == True) and (merge_old_cluster):
        path = path + "Decay_MClus/"
    elif decay == True:
        path = path + "Decay/"
    if create_dir:
        try:
            os.makedirs(path)
            print("directory created [",path,"]")
        except Exception as ex:
            print("", ex)
    return path

def outputFileNameFormatter(resultDir, dataset, outputPrefix, ALPHA, BETA, LAMDA, decay, merge_old_cluster):
    output = ""
    path = output_directory_path(resultDir, dataset, outputPrefix, decay, merge_old_cluster)
    if decay == True:
        output = path + dataset + outputPrefix + "_ALPHA" + str(ALPHA) + "_BETA" + str(BETA) + "_LAMDA" + str(LAMDA) + ".txt"
    else:
        output = path + dataset + outputPrefix + "_ALPHA" + str(ALPHA) + "_BETA" + str(BETA) + ".txt"
    return output


def grid_search(args):
    _values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003,
                          0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                          0.09, 0.1, 0.2, 0.3, 0.4,0.5,0.4,0.6,0.7,0.8,0.9,1.0]
    alpha_list = []
    beta_list = []
    alpha_list.extend(_values)
    beta_list.extend(_values)
    if args['alpha_start_limit'] != -1:
        si = _values.index(float(args['alpha_start_limit']))
        if args['alpha_last_limit'] != -1:
            li = _values.index(args['alpha_last_limit'])
            alpha_list = _values[si:li+1]
        else:
            alpha_list = _values[si:]

    if args['beta_start_limit'] != -1:
        beta_start_index = _values.index(float(args['beta_start_limit']))
        if args['beta_last_limit'] != -1:
            beta_last_index = _values.index(args['beta_last_limit'])
            beta_list = _values[beta_start_index:beta_last_index + 1]
        else:
            beta_list = _values[beta_start_index:]

    alphas = []
    betas = []
    for a in alpha_list:
        for b in beta_list:
            alphas.append(a)
            betas.append(b)
    return alphas, betas

def handle_output_prefix(applyICF, applyCWW, FR_THRESHOLD = -1, single_term_clustering = False, local_cluster_beta = False, merge_old_clusters = False, EPM_batch = -1, EPM_samples = -1, EPM_interval = -1, applyWordSpecificity = False):
    outputPrefix = ""
    if merge_old_clusters:
        outputPrefix = outputPrefix + "_MOC"     # merge old clusters while decay weight reaches to zero
    if local_cluster_beta:
        outputPrefix = outputPrefix + "_LCB"     # beta calculation with cluster vocabulary, otherwise beta will be calculated by model active vocabulary
    if single_term_clustering:
        outputPrefix = outputPrefix + "_STC"     # single term must be matched if we want to calculate cluster probability
    if applyICF:
        outputPrefix = outputPrefix + "_ICF"     # inverse cluster frequency
    if applyWordSpecificity:
        outputPrefix = outputPrefix + "_WS"     # inverse cluster frequency
    if applyCWW:
        outputPrefix = outputPrefix + "_CWW"     # word-cooccurance
    if FR_THRESHOLD > 0:
        outputPrefix = outputPrefix + "_FR" + str(FR_THRESHOLD)
    if EPM_batch != -1:
        outputPrefix = outputPrefix + "_EPB" + str(EPM_batch)
        outputPrefix = outputPrefix + "_EPS" + str(EPM_samples)
        outputPrefix = outputPrefix + "_EPI" + str(EPM_interval)


    return outputPrefix


def argument_parser():
    ap = argparse.ArgumentParser(description='-d \"News\" -o \"result/\" -icf -cww -decay 0.000006 -alpha 0.002 -beta 0.0004')
    ap.add_argument("-d", "--dataset_dir", required=True,  help="the dataset file path. file format should be json on each line of file { Id: '' , clusterNo: '', textCleaned:'' }")
    ap.add_argument("-o", "--output_dir", required=True,  help="[the directory where the output will be saved] ")
    ap.add_argument("-gs", "--generate_summary", default=False, action='store_true', required=False,
                    help="True/False [if you want to generate summary of generated results]")
    ap.add_argument("-icf", "--icf", default=False, action='store_true', required=False, help="True/False [if you want to apply ICF weight]")
    ap.add_argument("-ws", "--ws", default=False, action='store_true', required=False, help="True/False [if you want to apply Word Specificity weight]")
    ap.add_argument("-cww", "--cww", default=False, action='store_true', required=False,  help="True/False [if you want to apply CWW weight]")
    ap.add_argument("-alpha", "--alpha", default=False, required=False, help="customized value(s) of alpha [if more than one, seperated by comma]")
    ap.add_argument("-beta", "--beta",  default=False, required=False, help="customized value(s) of beta [if more than one, seperated by comma]")
    ap.add_argument("-decay", "--decay", default=False, type=float, required=False, help="Default value is [False]. Value is set as lambda in for exponential decay i.e. 0.000006")
    ap.add_argument("-ft", "--feature_threshold", default=-1, type=int, required=False, help="triangular weight feature threshold")
    ap.add_argument("-sa", "--start_alpha", default=0, required=False, help="start alpha when we are doing grid search")
    ap.add_argument("-sb", "--start_beta", default=0, required=False, help="start beta when we are doing grid search")
    ap.add_argument("-asl", "--alpha_start_limit", default=0.0001, required=False, help="start alpha from grid search array ")
    ap.add_argument("-bsl", "--beta_start_limit", default=0.0001, required=False, help="start beta from grid search array")
    ap.add_argument("-all", "--alpha_last_limit", default=0.9, required=False,help="end alpha from grid search array")
    ap.add_argument("-bll", "--beta_last_limit", default=0.9, required=False, help="end beta from grid search array")
    ap.add_argument("-log", "--log_file", default=False, required=False, help="log file flag ")
    ap.add_argument("-stc", "--single_term_consider", default=False, action='store_true', required=False, help="True/False [Atleast one term should be matched before calculating cluster probability]")
    ap.add_argument("-lastindex", "--last_index_for_grid_search", default=0, type=int, required=False,
                    help="start index while doing grid search")
    ap.add_argument("-lcb", "--local_cluster_beta", default=False, action='store_true', required=False, help="True -[beta will be calculated according to cluster vocabulary] ")
    ap.add_argument("-mclus", "--merge_old_cluster", default=False, action='store_true', required=False,
                    help="True/False [if you want to merge deleted cluster]")


    ap.add_argument("-epb", "--episodic_memory_batch", default=-1, type=int, required=False,
                    help="Episodic memory replay batch size, -1 is not applied, value > 0 ")
    ap.add_argument("-eps", "--ep_random_samples", default=30, type=int, required=False,
                    help="Episodic memory replay instances , -epb > value > 0 ")

    ap.add_argument("-epi", "--ep_interval", default=100, type=int, required=False,
                    help="Episodic memory replay interval, value > 0 ")

    args = vars(ap.parse_args())
    for k, v in args.items():
        print(k, " -> ",v)
    return args

def grid_search_parameter(list_of_values):
    alphas = []
    betas = []
    for v in list_of_values:
        alphas.extend(list_of_values)
    for v in list_of_values:
        betas .extend([v for x in range(0, len(list_of_values))])
    return alphas, betas

if __name__ == '__main__':
    args = argument_parser()
    # print("Setting this process in High Priority")
    # p = psutil.Process(os.getpid())
    # p.nice(psutil.HIGH_PRIORITY_CLASS)

    resultDir = "result/"
    datasets = ["News", "News-T", "Tweets", "reuters21578", "Tweets-T", "reuters21578-T"]
    dataset = "data/"+datasets[0]

    alphas, betas = grid_search(args)
    LAMDA = 0.000006
    batch_size = 2000
    feature_threshold = -1

    decay = False
    applyICF = False
    applyCWW = False
    stc = False

    # -------------------Argument Handling----------------------------------
    start_alpha = float(args['start_alpha'])
    start_beta = float(args['start_beta'])
    dataset = args['dataset_dir']
    resultDir = args['output_dir']

    stc = args['single_term_consider']
    applyICF = args['icf']
    applyWordSpecificity = args['ws']
    applyCWW = args['cww']
    alpha_param = args['alpha']
    if alpha_param != False:
        alphas = [float(x) for x in alpha_param.split(",")]
    beta_param = args['beta']
    if beta_param != False:
        betas = [float(x) for x in beta_param.split(",")]
    decay_param = args['decay']
    if decay_param != False:
        decay = True
        LAMDA = float(decay_param)

    epi_mem_batch = args['episodic_memory_batch']
    epi_mem_interval = args['ep_interval']
    epi_mem_samples = args['ep_random_samples']

    feature_threshold = int(args['feature_threshold'])

    start_index_for_grid_search = args['last_index_for_grid_search']
    if start_index_for_grid_search > 0:
        start_alpha = alphas[start_index_for_grid_search]
        start_beta = betas[start_index_for_grid_search]

    local_cluster_vocabulary_beta = args['local_cluster_beta']
    outputPrefix = handle_output_prefix(applyICF=applyICF, applyCWW=applyCWW, FR_THRESHOLD=feature_threshold, single_term_clustering=stc, local_cluster_beta = local_cluster_vocabulary_beta, EPM_batch=epi_mem_batch, EPM_samples=epi_mem_samples, EPM_interval=epi_mem_interval, applyWordSpecificity=applyWordSpecificity)
    merge_old_cluster = args['merge_old_cluster']
    # ----------------------Argument Handler ----------------------------

    start_time = time.time()
    print("Dataset: ",dataset," , Decay:", decay, " , ICF = ", applyICF, " , CWW = ", applyCWW)
    listOfObjects = []
    # reading instances from dataset file and store in list-----------
    with open(dataset) as input:
        line = input.readline()
        while line:
            obj = json.loads(line)  # a line is a document represented in JSON
            listOfObjects.append(obj)
            line = input.readline()
    #------------------- reading fininshed ---------------
    printExecutionTime(start_time)
    start_time = time.time()

    total_iteration = alphas.__len__()
    idx = -1
    outputPath = output_directory_path(resultDir, ntpath.basename(dataset), outputPrefix, decay, True)
    for a,b in zip(alphas, betas):
        idx += 1
        ALPHA = a
        BETA = b
        if start_alpha != 0 and start_beta != 0:
            if ALPHA != start_alpha:
                continue
            elif BETA != start_beta:
                continue
            else:
                start_alpha = 0
                start_beta = 0

        print(total_iteration , "/" , (idx + 1) , "  ~ APLHA " , (ALPHA) , " -  BETA " , (BETA))
        output = outputFileNameFormatter(resultDir, ntpath.basename(dataset), outputPrefix, ALPHA, BETA, LAMDA, decay, merge_old_cluster)

        episodic_memory = list()

        model = Model(ALPHA, BETA, LAMDA, applyDecay=decay, applyICF = applyICF, applyCWW=applyCWW, single_term_clustering = stc, FR_THRESHOLD = feature_threshold, local_vocabulary_beta = local_cluster_vocabulary_beta, merge_old_cluster = merge_old_cluster, applyWordSpecificity=applyWordSpecificity)
        iter = 1
        # batch_documents = []
        for obj in listOfObjects:
            document = Document(obj, model.words.word_wid_map, model.words.wid_word_map,
                                model.wid_docId, model.word_counter)  # creating a document object which will spilt the text and update wordToIdMap, wordList



            if iter%batch_size == 0:
                start_time=printExecutionTime(start_time,"Documents "+str(iter))
                # model.gibbs_sampling(batch_documents)
                # model.NEWG(batch_documents)

                # batch_documents = []

            # batch_documents.append(document)
            sampled_cluster_id = model.processDocument(document)
            iter += 1

            # For episodic memory
            if epi_mem_batch != -1:
                if episodic_memory.__len__() >= epi_mem_batch:
                    del episodic_memory[0]
                document.cluster_id = sampled_cluster_id
                episodic_memory.append(document)

                if iter % epi_mem_interval == 0 and episodic_memory.__len__() == epi_mem_batch:
                    random_docs = sample(episodic_memory, epi_mem_samples)# choose random samples

                    for doc in random_docs:
                        try:
                            model.remove_docoment_from_cluster_feature(doc, doc.cluster_id)
                            new_cluster_id = model.sampleCluster(doc)
                            doc.cluster_id = new_cluster_id
                        except KeyError as ex:
                            print(repr(ex))
                            print(str(doc))
                            exit(-1)
                    # remove from current clusters
                    # sample again


        # Printing Clusters into File
        print_dic("dic.data",model.words.wid_word_map)
        f = open(output, "w")
        for d in model.docIdClusId:
            st = ""+str(d)+" "+str(model.docIdClusId[d])+" \n"
            f.write(st)
        for d in model.deletedDocIdClusId:
            st = ""+str(d)+" "+str(model.deletedDocIdClusId[d])+" \n"
            f.write(st)
        f.close()
        current_time = time.strftime("%H:%M:%S [%Y-%m-%d]", time.localtime())
        print(output, current_time)
        printExecutionTime(start_time)

        # end of beta loop
    #end of alpha loop

    if (args['generate_summary']):
        evaluate_results(dataset, outputPath, "stats/")

    printExecutionTime(start_time)

