# import modules
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import ast
import re
from nltk.tokenize import RegexpTokenizer
import gensim.models
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Following lines commented out for running program in VScode
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib','inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # for plot styling

###############################################################################

# Creating the Word2Vec Model:

# store file path for ease of use
file_path = "/Users/zanemazorbrown/Desktop/reviews_Patio_Lawn_and_Garden_5.json"

# create empty list to append to
review_list = []

# open dataset document and store review data in review_list
with open(file_path, "r") as infile:
    for line in infile.readlines():
        review = ast.literal_eval(line)
        review_list.append(review)

# Define function to remove html tags


def remove_html_tags(text):
    p = re.compile('<.*?>')
    return p.sub(' ', text)


# Contraction dictionary from: https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
                    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not",
                    "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                    "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                    "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                    "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                    "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                    "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                    "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                    "you'll've": "you will have", "you're": "you are", "you've": "you have"}


def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


contractions, contractions_re = _get_contractions(contraction_dict)

# define function to deconstruct contractions


def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# define function to tokenize review text


def tokenize(text):
    tokenizer = RegexpTokenizer("[\w']+")
    return tokenizer.tokenize(text)

# define function to remove stopwords from tokenized text
# Retrieved list form of stopwords from a comment under the following github:
# https://gist.github.com/sebleier/554280


def rv_stopwords(tokenized_text):
    sw_list = [",", ".", "'", '"', "i", "me", "my", "myself", "we", "our",
               "ours", "ourselves", "you", "your", "yours", "yourself",
               "yourselves", "he", "him", "his", "himself", "she", "her",
               "hers", "herself", "it", "its", "itself", "they", "them",
               "their", "theirs", "themselves", "what", "which", "who",
               "whom", "this", "that", "these", "those", "am", "is", "are",
               "was", "were", "be", "been", "being", "have", "has", "had",
               "having", "do", "does", "did", "doing", "a", "an", "the", "and",
               "but", "if", "or", "because", "as", "until", "while", "of",
               "at", "by", "for", "with", "about", "against", "between",
               "into", "through", "during", "before", "after", "above",
               "below", "to", "from", "up", "down", "in", "out", "on", "off",
               "over", "under", "again", "further", "then", "once", "here",
               "there", "when", "where", "why", "how", "all", "any", "both",
               "each", "few", "more", "most", "other", "some", "such", "no",
               "nor", "not", "only", "own", "same", "so", "than", "too",
               "very", "s", "t", "can", "will", "just", "don", "should",
               "now"]
    return [word for word in tokenized_text if word not in sw_list]

# preprocess function combining all preprocessing measures


def preprocess(text):
    text = remove_html_tags(text)
    text = replace_contractions(text)
    text = text.lower()
    tokens = tokenize(text)
    tokens = rv_stopwords(tokens)
    return tokens


# create empty list to append to
corpus_raw = []

# store review text in a list for ease of use
for key in review_list:
    text = key["reviewText"]
    corpus_raw.append(text)

# create empty list to append to
corpus = []

# preprocess the review text and store in the corpus list
for idx, text in enumerate(corpus_raw):
    if(idx % 1000 == 0):
        print("{0} reviews have been preprocessed.".format(idx))
    corpus.append(preprocess(text))

# display message at the end of data preprocessing
print("Preprocessing Complete")

# create Word2Vec model using corpus
model = gensim.models.Word2Vec(
    sentences=corpus, sg=0, vector_size=100, window=5)

###############################################################################

# Supervised Method - Linear Regression (K-Fold Cross-Validation)

# create empty list to append MSE plot data to
line_chart_x = []
line_chart_y = []

# iterate through LinReg models and append MSEs for plotting/calculation
for i in range(5, 100):
    # Prepare input and output for linear regression
    X = []
    Y = []

    for t in model.wv.most_similar(positive=['amazing'], topn=150):
        X.append(model.wv[t[0]][:i])
        Y.append(t[1])
    for t in model.wv.most_similar(positive=['fertilizing'], topn=150):
        X.append(model.wv[t[0]][:i])
        Y.append(t[1])

    # convert X and Y to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    MSE = []
    # split dataset randomly
    kf = KFold(n_splits=10, shuffle=True, random_state=11)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        reg = LinearRegression().fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        MSE.append(metrics.mean_squared_error(Y_test, Y_pred))

    line_chart_x.append(i)
    line_chart_y.append(np.mean(MSE))

# display average MSE across i sample models
print(
    f"Average MSE is {np.mean(line_chart_y)} across {len(line_chart_x)} samples")
# visually represent MSE data
plt.plot(line_chart_x, line_chart_y)
plt.show()

###############################################################################

# Unsupervised Learning Model – Cluster Analysis

# prepare input for cluster analysis
word_list = []
wv_list = []

# cost related word "expensive"
for t in model.wv.most_similar(positive=['expensive'], topn=50):
    word_list.append(t[0])
    wv_list.append(model.wv[t[0]])
# animal related word "pests"
for t in model.wv.most_similar(positive=['pests'], topn=50):
    word_list.append(t[0])
    wv_list.append(model.wv[t[0]])
# product related word "plants"
for t in model.wv.most_similar(positive=['plants'], topn=50):
    word_list.append(t[0])
    wv_list.append(model.wv[t[0]])

# fit kmeans to the data and predict y_kmeans
wv_list = np.array(wv_list)
kmeans = KMeans(n_clusters=3)
kmeans.fit(wv_list)
y_kmeans = kmeans.predict(wv_list)

# use TSNE to reduce dimensions
num_dimensions = 2
tsne = TSNE(n_components=num_dimensions, random_state=0)
vectors = tsne.fit_transform(wv_list)

x_vals = [v[0] for v in vectors]
y_vals = [v[1] for v in vectors]

indices = list(range(len(word_list)))
selected_indices = random.sample(indices, 30)

# note: the following plot does not work outside of a Jupyter Notebook.
# a screenshot of the plot can be accessed using the following link:
# https://cpslo-my.sharepoint.com/:i:/g/personal/zmazorbr_calpoly_edu/ERL5TrIv9pBBiiJDCVQFJeEBujSF4a9-Rykk9YBinUqung?e=55jDa4
plt.figure(figsize=(12, 12))
for i in selected_indices:
    plt.annotate(word_list[i], (x_vals[i], y_vals[i]))

plt.scatter(x_vals, y_vals, c=y_kmeans, s=30, cmap='viridis')
