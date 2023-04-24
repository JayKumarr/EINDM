# EINDM

J. Kumar, J. Shao, R. Kumar, S. U. Din, C. B. Mawuli, and Q. Yang,  and , "A Context-enhanced Dirichlet Model for Online
Clustering in Short Text Streams," in Expert Systems with Applications, vol. XX, pp. xxx-xxx, Apr. 2023.

* Python 3.7.x:

# <h3>Example:</h3>

python main.py -d "data/Tweets-T-N" -o "results/" -ws -icf -stc -cww -lcb -mclus -alpha 0.04 -beta 5e-4 -decay 6e-6 -epb 500 -eps 30 -epi 60 -gs

# <h3>Parameters Definitions:</h3>
* -d  : dataset directory
* -ws  : include word specificity
* -icf  : include inverse cluster frequency
* -stc : calculate probability for cluster containing at least one common term btw doc and cluster
* -cww   : include word-to-word co-occurrence probability
* -lcb   : include cluster-based beta value
* -mclus : enable merging of outdated clusters
* -alpha  : Alpha concentration parameter
* -beta  : beta parameter
* -decay  : exponential decay
* -epb  : queue size
* -eps  : episodic inference samples
* -epi  : episodic inference interval
* -gs  : generate summary
