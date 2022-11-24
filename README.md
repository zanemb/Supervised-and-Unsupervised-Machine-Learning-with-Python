# Supervised-and-Unsupervised-Machine-Learning-with-Python
Performed supervised (Linear Regression) and unsupervised (Cluster Analysis) machine learning techniques on 13,272 reviews for Patio, Lawn, and Garden products from amazon.com. The raw review data is preprocessed to remove any stray html tags, find and replace any contractions, and is tokenized before creating the Word2Vec model, which is used as input for both ML techniques. The dataset can be found at https://jmcauley.ucsd.edu/data/amazon/ in the K-cores column.

# Linear Regression (K-Fold Cross-Validation)
I used a K value of 10 so as not to overfit the data (as there is over 13,000 reviews in the dataset). This allows the model to produce more generalized results, as opposed to simply memorizing the patterns in the training set. The shuffle and random_state arguments were also used to increase randomization of dataset splits.
Because the data is split randomly for each sample, we will rarely see the same MSE twice. Across a sample set of 95 MSEs, calculated as the averages of each set of 10 k-folds, the average MSE trends between 0.003 and 0.004. This is a relatively low MSE range, meaning this model has good predictive performance against the testing data.

# Cluster Analysis
I created three clusters using the keywords "expensive," "pests," and "plants." Each of these words represent a relevant section of the "Patio, Lawn, and Garden" domain. This combination of words specifically produced three distinct clusters with little overlap. While the clusters are not perfect, each cluster is clearly separated from the other two. 
For best results in viewing the cluster plot, uncomment out the get_ipython() lines and run using Jupyter. 
