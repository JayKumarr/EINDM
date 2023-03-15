
K_DOC_ID = "Id"
K_DOC_TIME = "time"
K_DOC_TEXT = "textCleaned"
K_CLASS_ID = "clusterNo"

I_cn = 0
I_cwf = 1     # cluster word frequency
I_cww = 2     # cluster word to word co-occurence
I_cd = 3      # cluster decay
I_csw = 4     # total number of words in cluster
I_cl = 5      # updated time stamp
I_pd = 6   # previously deleted, extension if the cluster updated after first time deletion
I_CWORD_ARRIVAL_TIME = 7
I_CTIME = 8
# variables which have underscore '_'  are globaly shareable and updatable
# variable without underscore are local variables

#------ Variable defined Terminology
# wid      :   word ID
# word     :  a term of word in text
# clus     :  Cluster
# freq     :  frequency
# docId    :  Document ID


#-------Cluster Feature Representation
# cn   : number of documents in cluster , documents indexs
# cwf  : word frequency in cluster          {wid, frequency}
# cww  : word to word co-occurance matrix
# cd   : cluster decay weight
# csw  : number of words in cluster ,  sum of all frequencies of words
# cl   : last time stamp