from numpy.__init__ import str_

from Document import Document
import Contant as con
import math
from Words import Words
from PrintingFile import print_dic
from  Cluster import Cluster

class Model:

    def __init__(self, ALPHA, BETA, LAMDA, applyDecay=True, applyICF = True, applyCWW = False, single_term_clustering = False, FR_THRESHOLD=-1, local_vocabulary_beta = False, merge_old_cluster = False, applyWordSpecificity = False ):
        self.alpha = ALPHA
        self.beta = BETA
        self.applyDecay = applyDecay
        self.applyICF = applyICF
        self.applyWS = applyWordSpecificity
        self.applyCWW = applyCWW
        self.atleast_one_term_matched_for_clustering = single_term_clustering
        self.applyFeatureReduction = False
        if (FR_THRESHOLD >0):
            self.applyFeatureReduction = True
        self.feature_threshold = FR_THRESHOLD
        self.local_cluster_vocabulary_beta  = local_vocabulary_beta     # if we want to calculate beta according to related cluster vocabulary. if not it will calculate the model active vocabulary beta
        self.merge_old_cluster = merge_old_cluster

        self.words = Words()
        self.wid_docId = {}      # wordID, documentId:    updated by Document
        self.active_clusters = {}   #   clusterID -> [ cn, cw, cwf, cww, cd, csw]
        self.active_documents = {}   # {documentId, Document}
        self.widClusid = {}   # {wordID ,clusterID }: to know that how many cluster this word has occured
        self.docIdClusId = {}  # {documentID , clusterID} Cluster assignments of each document
        self.deletedDocIdClusId = {} # those documents which are deleted while deleting the cluster, #this DS will be utilized to print output
        self.lamda = LAMDA

        self.word_counter = {0:0}
        self.cluster_counter = {0:0}

        self.currentTimestamp = 0

        # self.tmp_wid_neighbor_file = open('wid_neighbor_file_News.txt','w')
        # self.tmp_widtf_neighbors_file = open('widTF_neighbors_file_News.txt','w')

    def processDocument(self, document):
        self.active_documents[document.docId] = document
        self.currentTimestamp+=1
        if (self.applyDecay == True):
            self.checkOldClusters(self.lamda)
        sampled_cluster_id = self.sampleCluster(document)
        if (self.applyFeatureReduction):
            for clus_id, CF in self.active_clusters.items():
                self.check_cluster_outdated_features(clus_id, CF, self.feature_threshold)  # UPDATED LINE

        # self.print_termtf_neighbors_in_file(self.tmp_widtf_neighbors_file)
        # self.print_term_neighbors_in_file(self.tmp_wid_neighbor_file)

        return sampled_cluster_id

    def sampleCluster(self, document):

        clusIdOfMaxProb = -1
        clusMaxProb = 0.0

        N = self.active_documents.__len__()  # number of maintained documents, some documents might be deleted from cluster
        VintoBETA = self.getVocabularyIntoBeta()
        beta_sum = 0.0
        count_related_clusters = 0

        # need to calculate probablity of existing clusters, if no existing cluster this loop will be skipped
        for clusId in self.active_clusters:
            CF = self.active_clusters[clusId]

            cluster_wids = CF[con.I_cwf].keys()
            doc_wids = document.widFreq.keys()
            common_wids = intersection(cluster_wids, doc_wids)
            if (self.atleast_one_term_matched_for_clustering):

                if common_wids.__len__() < 1:
                    continue

                # uncommon_words_of_doc = set_substraction(doc_wids, cluster_wids)
                # uncommon_words_of_clus = set_substraction(cluster_wids, doc_wids)
                # composite_vocabulary = set_union(cluster_wids, doc_wids)
            # --- updation for beta calculation
            if (self.local_cluster_vocabulary_beta):
                VintoBETA = float(self.beta)*float(CF[con.I_cwf].__len__())
                beta_sum+=VintoBETA
                count_related_clusters+=1

            numOfDocInClus = CF[con.I_cn].__len__()
            eqPart1 = float(numOfDocInClus) / float(( N-1 + self.alpha*N))
            eqPart2Nominator = 1.0
            eqPart2Denominator = 1.0   # usually it should be zero
            numOfWordsInClus = CF[con.I_csw]
            i = 0  # represent word count in document
            for w in document.widFreq:
                widFreqInClus = 0
                if w in CF[con.I_cwf]:  # if the word of the document exists in cluster
                    widFreqInClus = CF[con.I_cwf][w]

                icf = 1.0
                ws = 1.0
                if (self.applyICF == True):  # This condition is used to control parameters by main method
                    icf = self.ICF(w)

                if (self.applyWS == True):  # This condition is used to control parameters by main method
                    ws = self.word_specificity(w)


                freq = document.widFreq[w]
                for j in range(freq):
                    i += 1
                    eqPart2Nominator *= (widFreqInClus * icf *ws + self.beta + j)
                    eqPart2Denominator *= (numOfWordsInClus + VintoBETA + i)

            eqPart2 = eqPart2Nominator / eqPart2Denominator
            if self.applyCWW:  # to control applying CWW from main method
                eqPart2 = eqPart2 * self.addingWidToWidWeightInEqPart2(document, CF, eqPart2)

            clusProb = eqPart1 * eqPart2
            if clusProb > clusMaxProb:
                clusMaxProb = clusProb
                clusIdOfMaxProb = clusId

        # end for , all probablities of existing clusters have been calculated

        if (self.local_cluster_vocabulary_beta) and (count_related_clusters > 0):
            VintoBETA = float(beta_sum)/float(count_related_clusters)

        # need to calculate probablity of creating a new cluster
        eqPart1 = (self.alpha * N) / (N - 1 + self.alpha * N)
        eqPart2Nominator = 1.0
        eqPart2Denominator = 1.0
        i = 0 # represent word count in document
        for w in document.widFreq:
            freq = document.widFreq[w]

            for j in range(freq):
                i += 1
                eqPart2Nominator*= (self.beta+j)
                eqPart2Denominator*= (VintoBETA+i)
        probNewCluster = eqPart1*(eqPart2Nominator/eqPart2Denominator)
        if probNewCluster < clusMaxProb:
            self.addDocumentIntoClusterFeature(document, clusIdOfMaxProb)
            return clusIdOfMaxProb
        else:
            n_cluster_id = self.createNewCluster(document)
            return n_cluster_id



    def getVocabularyIntoBeta(self):
        temp = float(self.beta)*float(self.wid_docId.__len__())
        return temp

    def createNewCluster(self,document):
        #create new cluster
        self.cluster_counter[0] = self.cluster_counter[0]+1
        newIndexOfClus = self.cluster_counter[0] # = {}   clusterID -> [ cn, cw, cwf, cww, cd, csw]

        self.active_clusters[newIndexOfClus]={}
        self.active_clusters[newIndexOfClus][con.I_cn]=[]     # docs
        self.active_clusters[newIndexOfClus][con.I_cwf] = {}  # word frequency
        self.active_clusters[newIndexOfClus][con.I_cww] = {}  # word2word occurance
        self.active_clusters[newIndexOfClus][con.I_cd] = 1.0  # decay weight
        self.active_clusters[newIndexOfClus][con.I_csw] = 0   # total words
        self.active_clusters[newIndexOfClus][con.I_CWORD_ARRIVAL_TIME] = {}
        self.active_clusters[newIndexOfClus][con.I_CTIME] = self.currentTimestamp

        self.addDocumentIntoClusterFeature(document, newIndexOfClus)
        return newIndexOfClus

    def addDocumentIntoClusterFeature(self,document, clusterId):
        CF = self.active_clusters[clusterId]
        CF[con.I_cl] = self.currentTimestamp
        CF[con.I_cd] = 1.0
        self.docIdClusId[document.docId] = clusterId
        CF[con.I_cn].append(document.docId)
        # update feature of cluster
        for w in document.widFreq:
            self.updateWidClusid(w, clusterId)   #helps to calculate ICF, if this word is not contained by widClusMap then add it

            if w not in CF[con.I_cwf]:
                CF[con.I_cwf][w]=0
            if w not in CF[con.I_cww]:
                CF[con.I_cww][w] = {}

            CF[con.I_cwf][w] = CF[con.I_cwf][w] + document.widFreq[w]   #update word frequency in cluster
            CF[con.I_csw] = CF[con.I_csw]+document.widFreq[w]           # increasing number of words in cluster

            if (self.applyFeatureReduction):  # if true then maintain term arrival time
                # update arrival time of wid
                if w not in CF[con.I_CWORD_ARRIVAL_TIME]:
                    CF[con.I_CWORD_ARRIVAL_TIME][w] = []
                CF[con.I_CWORD_ARRIVAL_TIME][w].append(CF[con.I_cn].__len__())


            for w2 in document.widToWidFreq[w]: #updating CF[cww] word to word frequency
                if w!=w2:
                    if w2 not in CF[con.I_cww][w]:
                        CF[con.I_cww][w][w2] = document.widToWidFreq[w][w2]
                    else:
                        CF[con.I_cww][w][w2] = CF[con.I_cww][w][w2]+document.widToWidFreq[w][w2]




    def remove_docoment_from_cluster_feature(self, document, clusterId):
        CF = self.active_clusters[clusterId]
        cluster_docs = CF[con.I_cn]
        CF[con.I_cn].remove(document.docId)
        # need to code for word arrival time removing
        if False: #cluster_docs.__len__() == 1:
            print(str(document), " -> deleting:", clusterId)
            self.deleteOldCluster(clusterId, CF)
        else:
            for wid in document.widFreq:
                CF[con.I_cwf][wid] = CF[con.I_cwf][wid] - document.widFreq[wid]  # update word frequency in cluster
                CF[con.I_csw] = CF[con.I_csw] - document.widFreq[wid]  # increasing number of words in cluster

                if CF[con.I_cwf][wid] == 0:
                    self.widClusid[wid].remove(clusterId)
                    if self.widClusid[wid].__len__() == 0:
                        del self.widClusid[wid]
                    del CF[con.I_cwf][wid]


                for w2 in document.widToWidFreq[wid]:  # updating CF[cww] word to word frequency
                    if wid != w2:
                        new_value = CF[con.I_cww][wid][w2] - document.widToWidFreq[wid][w2]
                        if new_value !=0:
                            CF[con.I_cww][wid][w2] = new_value
                        else:
                            del CF[con.I_cww][wid][w2]
                            if CF[con.I_cww][wid].__len__() == 0:
                                del CF[con.I_cww][wid]









    def updateWidClusid(self, wid, clusterId):
        if wid not in self.widClusid:  # updating widClusid
            self.widClusid[wid] = []
            self.widClusid[wid].append(clusterId)
        else:
            if clusterId not in self.widClusid[wid]:
                self.widClusid[wid].append(clusterId)


    def addingWidToWidWeightInEqPart2(self,document, CF, eqPart2):
        product = 1.0
        traversed = []
        for wid in document.widToWidFreq:
            if wid not in CF[con.I_cww]:  # if this word not exist in the cluster
                continue
            sumOfProbablitiesOfWid = 0.0
            for wid2 in document.widToWidFreq[wid]:
                sumOfProbablitiesOfWid = sumOfProbablitiesOfWid+document.widToWidFreq[wid][wid2]
            for wid2 in document.widToWidFreq[wid]:
                if wid2 in CF[con.I_cww][wid]:
                    if wid2 not in traversed:
                        weight = CF[con.I_cww][wid][wid2] / sumOfProbablitiesOfWid
                        product = product+weight
            traversed.append(wid)
        return product


    def checkOldClusters(self, LAMDA):
        threshold = 0.00001
        clustersToDelete = {}
        for clusterID in self.active_clusters:
            CF = self.active_clusters[clusterID]

            lastupdated = CF[con.I_cl]
            power = -LAMDA*(self.currentTimestamp-lastupdated)
            decay=pow(2,power)
            CF[con.I_cd] = CF[con.I_cd]*decay
            if CF[con.I_cd] < threshold:
                clustersToDelete[clusterID] = CF
        for clusIDKey, CFvalue in clustersToDelete.items():
            if (self.merge_old_cluster): #merge_old_cluster
                id = self.check_cluster_to_merge(CFvalue,clusIDKey)
                if id != clusIDKey:
                    self.merger_clusters(clusIDKey, id)
                    del[self.active_clusters[clusIDKey]]
                else:
                    self.deleteOldCluster(clusIDKey, CFvalue)
            else:
                self.deleteOldCluster(clusIDKey, CFvalue)


    def deleteOldCluster(self, clusterID, CF):

        for wid in CF[con.I_cwf]:  # remove words from self.widClusid
            self.widClusid[wid].remove(clusterID)
            if self.widClusid[wid].__len__() == 0:
                del[self.widClusid[wid]]
            listOfDocsContainsWid = self.wid_docId[wid]
            listOfDocToDelete=intersection(listOfDocsContainsWid, CF[con.I_cn])
            for docIdToDelete in listOfDocToDelete:
                self.wid_docId[wid].remove(docIdToDelete)
                if self.wid_docId[wid].__len__() == 0: #if a word is not used by any document then delete it
                    del[self.wid_docId[wid]]
                    word = self.words.wid_word_map[wid]
                    del[self.words.wid_word_map[wid]]
                    del[self.words.word_wid_map[word]]
        for docId in CF[con.I_cn]: # remove documents from self.active_documents, self.docIdClusId
            del[self.active_documents[docId]]
            del[self.docIdClusId[docId]]
            self.deletedDocIdClusId[docId] = clusterID #this DS will be utilized to print output
        del[self.active_clusters[clusterID]]



    def ICF(self,wid):
        icf = 1.0
        if self.active_clusters.__len__() < 5:
            icf = 1.0
        else:
            if wid in self.widClusid:
                icf = math.log2( self.active_clusters.__len__()/self.widClusid[wid].__len__())
        return icf

    def word_specificity(self, wid):
        if wid not in self.widClusid:
            return 1
        clusterIds=self.widClusid[wid]
        total_tf = 0
        unique_set_wid = set()
        for cluster_id in clusterIds:
            CF = self.active_clusters[cluster_id]
            total_tf = total_tf+CF[con.I_cwf][wid]
            if wid not in CF[con.I_cww]:
                continue
            neighbor_ids = CF[con.I_cww][wid].keys()
            unique_set_wid.update(neighbor_ids)

        neighbors = unique_set_wid.__len__()
        return customized_log_function( (neighbors/total_tf) )





    def print_term_neighbors_in_file(self, tmp_wid_neighbor_file):
        # 1) get all cluster containing this wid
        # 2) count number of unique neighbors
        # 3) count occurrence of term
        str_concat = "{"
        for wid, clusterIds in self.widClusid.items():
            unique_set_wid = set()
            for cluster_id in clusterIds:
                CF = self.active_clusters[cluster_id]
                if wid not in CF[con.I_cww]:
                    continue
                neighbor_ids = CF[con.I_cww][wid].keys()
                unique_set_wid.update(neighbor_ids)
            str_concat = str_concat + ","+ str(wid) +": "+str(unique_set_wid.__len__())+""
        str_concat = str_concat + "}"
        tmp_wid_neighbor_file.write(str_concat)
        tmp_wid_neighbor_file.write("\n")

    def print_termtf_neighbors_in_file(self, tmp_tf_neighbors_file):
        str_concat = "{"
        for wid, clusterIds in self.widClusid.items():
            unique_set_wid = set()
            total_term_frequency_of_wid = 0
            for cluster_id in clusterIds:
                CF = self.active_clusters[cluster_id]
                if wid not in CF[con.I_cww]:
                    continue
                neighbor_ids = CF[con.I_cww][wid].keys()
                unique_set_wid.update(neighbor_ids)
                total_term_frequency_of_wid += CF[con.I_cwf][wid]
            str_concat = str_concat + "," + str( total_term_frequency_of_wid ) + ": " + str(unique_set_wid.__len__()) + ""
        str_concat = str_concat + "}"
        tmp_tf_neighbors_file.write(str_concat)
        tmp_tf_neighbors_file.write("\n")


    def term_importance(self, document):  # this will derive the term importance with respect to term frequecy
        # 1) fetch all the clusters related to words of document
        # 2) sum all term frequency of each word in cluster
        # 3) percentage of each term of document in clusters
        # 4) use that percentage of each term to assign in cluster

        print("Hello")

    def NEWG(self, batch_documents):
        wid_freq = {}
        wid_wid_freq = {}
        for doc in batch_documents:
            for wid, freq in doc.widFreq.items():
                tf=wid_freq.get(wid,0) # zero is default
                wid_freq[wid] = tf + freq
                list_of_words = doc.widToWidFreq[wid]
                for cooccured_wids in list_of_words:
                    edge_score = doc.widToWidFreq[wid][cooccured_wids]
                    try:
                        wid_wid_freq[wid][cooccured_wids] = wid_wid_freq[wid][cooccured_wids] + edge_score
                    except:
                        wid_wid_freq[wid] = {}
                        wid_wid_freq[wid][cooccured_wids] = edge_score

        print_dic("wid_stats.data", wid_freq)

    # this function does not need changing
    def calculate_triangular_time(self, timestamp):
        return (( (timestamp*timestamp) + timestamp )/2)

    # this function does not need changing, calculate recency of terms according to cluster documents
    def check_cluster_outdated_features(self, clusterID, CF, FEATURE_RECENCY_THRESHOLD):
        # CF = self.clusters[cluster_id]
        wid_to_be_removed = []
        cluster_triangular_time = self.calculate_triangular_time(1)
        current_cluster_triangular_time = self.calculate_triangular_time(CF[con.I_cn].__len__())
        real_triangular_number =current_cluster_triangular_time - cluster_triangular_time   + 1
        for w_id in CF[con.I_cwf].keys():
            list_of_time_stamps = CF[con.I_CWORD_ARRIVAL_TIME][w_id]  # sequential number of document in cluster [1,2,3,4,5,6,7.....]
            word_actual_time_values = sum(list_of_time_stamps)
            recency = ((word_actual_time_values*100)/real_triangular_number)
            if recency < FEATURE_RECENCY_THRESHOLD:
                wid_to_be_removed.append(w_id)

        if wid_to_be_removed.__len__() > 0:  # update co-occurance of related wid
            update_coorrence_mtrix_according_to_cluster_features(CF,wid_to_be_removed)

        for wid in wid_to_be_removed:
            self.widClusid[wid].remove(clusterID)
            if self.widClusid[wid].__len__() == 0:
                del [self.widClusid[wid]]

            listOfDocsContainsWid = self.wid_docId[wid]
            listOfDocToDelete=intersection(listOfDocsContainsWid, CF[con.I_cn])
            for docIdToDelete in listOfDocToDelete:
                self.wid_docId[wid].remove(docIdToDelete)
                if self.wid_docId[wid].__len__() == 0: #if a word is not used by any document then delete it
                    del[self.wid_docId[wid]]
                    word = self.words.wid_word_map[wid]
                    del[self.words.wid_word_map[wid]]
                    del[self.words.word_wid_map[word]]
            del[CF[con.I_cwf][wid]]  # deleting from cluster
            #  -------- ------- ------- -----

    def check_cluster_to_merge(self, cluster, cluster_id):
        document = Cluster(cluster)

        clusIdOfMaxProb = -1
        clusMaxProb = 0.0

        N = self.active_documents.__len__()  # number of maintained documents, some documents might be deleted from cluster
        VintoBETA = self.getVocabularyIntoBeta()
        beta_sum = 0.0
        count_related_clusters = 0

        # need to calculate probablity of existing clusters, if no existing cluster this loop will be skipped
        for clusId in self.active_clusters:
            CF = self.active_clusters[clusId]

            cluster_wids = CF[con.I_cwf].keys()
            doc_wids = document.widFreq.keys()
            common_wids = intersection(cluster_wids, doc_wids)
            if common_wids.__len__() < 1:
                continue

            # --- updation for beta calculation
            if (self.local_cluster_vocabulary_beta):
                # VintoBETA = float(self.beta) * float( union(CF[con.I_cwf].keys(), document.widFreq.keys()).__len__() ) # combine vocabulary of both cluster and document to calculate local beta
                VintoBETA = float(self.beta) * float(CF[con.I_cwf].__len__()) # consider cluster vocabulary to compute beta
                beta_sum += VintoBETA
                count_related_clusters += 1

            numOfDocInClus = CF[con.I_cn].__len__()
            eqPart1 = float(numOfDocInClus) / float((N - 1 + self.alpha * N))
            eqPart2Nominator = 1.0
            eqPart2Denominator = 1.0
            numOfWordsInClus = CF[con.I_csw]
            i = 0  # represent word count in document
            for w in document.widFreq:
                widFreqInClus = 0
                if w in CF[con.I_cwf]:  # if the word of the document exists in cluster
                    widFreqInClus = CF[con.I_cwf][w]

                icf = 1.0
                if (self.applyICF == True):  # This condition is used to control parameters by main method
                    icf = self.ICF(w)

                freq = document.widFreq[w]
                for j in range(freq):
                    i += 1
                    eqPart2Nominator *= (widFreqInClus * icf + self.beta + j)
                    eqPart2Denominator *= (numOfWordsInClus * VintoBETA + i)

            eqPart2 = eqPart2Nominator / eqPart2Denominator
            if (self.applyCWW == True):  # to control applying CWW from main method
                eqPart2 = eqPart2 * self.addingWidToWidWeightInEqPart2(document, CF, eqPart2)

            clusProb = eqPart1 * eqPart2
            if clusProb > clusMaxProb:
                clusMaxProb = clusProb
                clusIdOfMaxProb = clusId
        # end for , all probablities of existing clusters have been calculated

        if (self.local_cluster_vocabulary_beta) and (count_related_clusters > 0):
            VintoBETA = float(beta_sum)/float(count_related_clusters)

        # need to calculate probablity of creating a new cluster
        eqPart1 = (self.alpha * N) / (N - 1 + self.alpha * N)
        eqPart2Nominator = 1.0
        eqPart2Denominator = 1.0
        i = 0 # represent word count in document
        for w in document.widFreq:
            freq = document.widFreq[w]

            for j in range(freq):
                i += 1
                eqPart2Nominator*= (self.beta+j)
                eqPart2Denominator*= (VintoBETA+i)
        probNewCluster = eqPart1*(eqPart2Nominator/eqPart2Denominator)
        if probNewCluster < clusMaxProb:
            return clusIdOfMaxProb
        else:
            return cluster_id

    def merger_clusters(self, cluster_id, clusIdOfMaxProb):
        CF_to_be_merged = self.active_clusters[cluster_id]
        CF = self.active_clusters[clusIdOfMaxProb]

        for docId in CF_to_be_merged[con.I_cn]:
            self.docIdClusId[docId] = clusIdOfMaxProb
            CF[con.I_cn].append(docId)

        for wid, w_freq in CF_to_be_merged[con.I_cwf].items():
            #helps to calculate ICF, if this word is not contained by widClusMap then add it
            self.widClusid[wid].remove(cluster_id)
            if clusIdOfMaxProb not in self.widClusid[wid]:
                self.widClusid[wid].append(clusIdOfMaxProb)

            if wid not in CF[con.I_cwf]:
                CF[con.I_cwf][wid]=0

            if wid not in CF[con.I_cww]:
                CF[con.I_cww][wid] = {}


            CF[con.I_cwf][wid] = CF[con.I_cwf][wid] + w_freq   #update word frequency in cluster
            CF[con.I_csw] = CF[con.I_csw] + w_freq           # increasing number of words in cluster


            if wid in CF_to_be_merged[con.I_cww]:
                for linked_w2 in CF_to_be_merged[con.I_cww][wid]: #updating CF[cww] word to word frequency
                    if linked_w2 not in CF[con.I_cww][wid]:
                        CF[con.I_cww][wid][linked_w2] = CF_to_be_merged[con.I_cww][wid][linked_w2]
                    else:
                        CF[con.I_cww][wid][linked_w2] = CF[con.I_cww][wid][linked_w2] + CF_to_be_merged[con.I_cww][wid][linked_w2]

            if (self.applyFeatureReduction):
                if wid not in CF[con.I_CWORD_ARRIVAL_TIME]:
                    CF[con.I_CWORD_ARRIVAL_TIME][wid] = []

                CF[con.I_CWORD_ARRIVAL_TIME][wid].extend(CF_to_be_merged[con.I_CWORD_ARRIVAL_TIME][wid])




def update_coorrence_mtrix_according_to_cluster_features( CF, removed_features_wids=[]): # this function will delete terms from co-occurence matrix [C_WW] which are not found in [C_WORD_FREQ]
    feature_set = CF[con.I_cwf].keys()
    features_coorrences = CF[con.I_cww].keys()
    if removed_features_wids.__len__() == 0:  # if user does not pass removed feature, then we have to create list of feature for deletion by looking at both matrix
        common_terms= intersection(feature_set,features_coorrences) # find active terms, not to be deleted
        if common_terms.__len__() == features_coorrences.__len__():   # if no wid for deletion
            return

        for wid_ww in features_coorrences: # traverse cooccurence matrix to find those terms which have to be deleted
            if wid_ww not in common_terms:
                removed_features_wids.append(wid_ww)
    for expired_wid in removed_features_wids:
        list_of_terms_coccured = CF[con.I_cww][expired_wid] # we have to remove expired term from other linked terms as well
        for linked_wid in list_of_terms_coccured:
            del[CF[con.I_cww][linked_wid][expired_wid]]
        del[CF[con.I_cww][expired_wid]]   # deleting expired term from C_WW


def intersection(listA, listB):
    return list(set(listA) & set(listB))

def set_substraction(listA, listB):
    return list(set(listA) - set(listB))

def set_union(listA, listB):
    return list(set(listA) | set(listB))

def union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def customized_log_function(value):

    # ex = (math.pow(2, (value*(value-3))))
    # eminusx = (math.pow(2, -(value*(value-3))))
    # v = ex / (ex+1)
    # return 1+v-0.2001

    ex = (math.pow(math.e, (value * (value - 3))))
    eminusx = (math.pow(math.e, -(value * (value - 3))))
    v = (ex - eminusx) / (ex + eminusx)
    return 2 + v

