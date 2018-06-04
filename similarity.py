import json
import spacy
import collections
import numpy as np
import re
from data_cursor import DataCursor
from matplotlib import cbook
from gensim.models import word2vec
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from scipy.spatial import distance
from sklearn.metrics import  pairwise 
from scipy.sparse import vstack
nlp = spacy.load('en_core_web_lg')
jfile = json.load(open('output/comments_out.json'))
from pandas import *

#workaround beacuse in this model they have no stop words
for word in nlp.Defaults.stop_words:
    lex = nlp.vocab[word]
    lex.is_stop = True


def get_caption(img_id):
	caption=""
	for post in jfile['results']:
		if (img_id==post['id']):
			caption= post['caption']
	return caption

def get_comment(img_id):
	comments=""
	for post in jfile['results']:
		if (img_id==post['id']):
			for comment in post['all_comments']:
				comment= comment['text']
				comments+=comment+" "
	return comments


def clean_text (text):
	text_complete=""
	doc_comment = nlp(unicode(text))
	string_text =''
	#print "DOC COMMENT --->", doc_comment
	for token_comment in doc_comment:
		token_comment = token_comment.lemma_
		doc_token = nlp(token_comment)
		token_comm=""
		for token in doc_token:
			if (token.is_stop==False):
				if (str(token) != '-PRON-'):
					token_comm=str(token)
				if (len(token_comm)>1):
					string_text+=token_comm+" "
	text_complete+=string_text
	text = re.sub(' +', " ", text_complete);
	return text
	

def comment_to_matrix(text):
	#word to vec by using spacy library and similarity function. 
	#similarity performed with GloVe which is "count-based" model
	text1=text.decode('utf-8')
	tokens = nlp(text1)
	matrix=[]
	for token1 in tokens:	
		token_result =[]
		for token2 in tokens:
			similarity=token1.similarity(token2)
			token_result.append(similarity)
		matrix.append(token_result)
	#print matrix
	return matrix
	
def caption_to_matrix(caption, comment):
	#word to vec by using spacy library and similarity function. 
	#similarity performed with GloVe which is "count-based" model
	caption_dec=caption.decode('utf-8')
	tokens_caption = nlp(caption_dec)
	comment_dec=comment.decode('utf-8')
	tokens_comment = nlp(comment_dec)
	
	matrix=[]
	for token_cap in tokens_caption:	
		token_result =[]
		for token_comm in tokens_comment:
			similarity=token_cap.similarity(token_comm)
			token_result.append(similarity)
		matrix.append(token_result)
	#print matrix
	return matrix

def sentence_to_matrix(text1,text2):
	doc1 = nlp(text1)
	doc2 = nlp(text2)
	similarity = doc1.similarity(doc2)
	return similarity

def cluster_comments(matrix_comment, nb_of_clusters, matrix_caption):
    #create k means object and pass to it the number of clusters
	kmeans = KMeans(nb_of_clusters)
	#pass the matrix to the fit method of kmeans to compute k-means clustering.
	kmeans.fit(matrix_comment)
	#y_kmeans = kmeans.predict(matrix_comment)
	#print "labels:", 
	kmeans.labels_
	clusters = collections.defaultdict(list)
	for i, label in enumerate(kmeans.labels_):
		clusters[label].append(i)
	#print "clusters:", clusters
	
	centers = kmeans.cluster_centers_
	#print "centers:",centers
	
	labels=kmeans.labels_
	
	prediction=kmeans.predict(matrix_caption)
	print "inertia", kmeans.inertia_
	#print "prediction", prediction

	return dict(clusters),labels, centers, prediction
	
	
def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
	distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
	return distances
	


def plot_cluster(matrix_comment,labels,predicted_label, centers,img_id, matrix_caption, caption):
	
	
	# append comment-caption-centers in same matrix -> than pass it do dim reduction
	plot_matrix = []
	matrix_comment=np.array(matrix_comment)
	comm_num_rows, comm_num_cols = matrix_comment.shape
	
	matrix_caption=np.array(matrix_caption)
	cap_num_rows, cap_num_cols = matrix_caption.shape
	
	centers=np.array(centers)
	center_num_rows, center_num_cols = centers.shape
	
	plot_matrix=vstack((matrix_comment,matrix_caption,centers))
	print "plot_matrix new dimension ->", plot_matrix.shape
	
	# convert two components as we're plotting points in a two-dimensional plane
	# "precomputed" because we provide a distance matrix
	# we will also specify `random_state` so the plot is reproducible.
	
	mds_model = TSNE(n_components=2,perplexity=40.0,metric='euclidean', random_state=1)
	dist_comm = 1 - cosine_similarity(plot_matrix)
	plot_matrix_embedded =mds_model.fit_transform(dist_comm)
	a= (np.array(plot_matrix_embedded)).shape
	
	
	
	cdict = {0: 'red', 1: 'blue', 2: 'green', 3:'dodgerblue', 4:'purple',5: 'orchid', 6: 'darkcyan', 7: 'pink', 8: 'yellow', 9: 'turquoise', 10: 'darkviolet',11:'orange'}
	scatter_x=plot_matrix_embedded[:comm_num_rows, 0]
	scatter_y=plot_matrix_embedded[:comm_num_rows, 1]
	
	"""
	fig, ax = plt.subplots()
	
	for l in np.unique(labels):
		ix = np.where(labels == l)
		ax.scatter(scatter_x, scatter_y, c = cdict[l], label = l, s = 50)
	ax.legend()

	#plot the caption value
	#add label in x,y position with the label as the film title
	
	dist_cap = 1 - cosine_similarity(matrix_caption)
	cap_embedded =mds_model.fit_transform(dist_cap)
	
	scatter_x1=cap_embedded[:, 0]
	scatter_y1=cap_embedded[:, 1]
	"""
	caption = caption.split()
	label_color = [cdict[l] for l in labels]
	plt.scatter(scatter_x, scatter_y, c=label_color)
	for i, caption in enumerate(caption_list):
		c=predicted_label[i]
		#plt.text(plot_matrix_embedded[54:59, 0],plot_matrix_embedded[54:59, 1], caption_list[i],color=cdict[c], size=14)
		center_num_rows=comm_num_rows+cap_num_rows
		print center_num_rows
		plt.scatter(plot_matrix_embedded[comm_num_rows:center_num_rows, 0],plot_matrix_embedded[comm_num_rows:center_num_rows, 1],  s=70, color=cdict[c], marker='+')
		#plt.scatter(scatter_x, scatter_y, s=10, c=cdict[c], alpha=0.5)
		
		plt.scatter(plot_matrix_embedded[center_num_rows:, 0],plot_matrix_embedded[center_num_rows:, 1], color='black', s=100, alpha=0.5)
		
	#plot the centroid
	
	
	
	
	
	#plt.savefig(str(img_id) + '.png')
	plt.show()

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    """
    norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms

        
"""clusters=km.fit_predict(data2D)
centroids = km.cluster_centers_

distances = []
for i, (cx, cy) in enumerate(centroids):
    mean_distance = k_mean_distance(data2D, cx, cy, i, clusters)
    distances.append(mean_distance)

print(distances)
dists = []
for row in matr:
    dists.append(scipy.spatial.distance.cosine(matr[0,:], row))
    
def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
        distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
        return distances
"""

if __name__ == "__main__":

	#get caption
	#clean caption
	for post in jfile['results']:
		img_id=post['id']
		print "Image id --- ", img_id
		caption = get_caption(img_id)
		comment = get_comment(img_id)

		#print "Comment --- ", comment
		comment_clean=clean_text(comment)
		#print "Comment clean --- ",comment_clean
		
		#print "Caption --- ", caption
		caption_clean=clean_text(caption)
		print "Caption cleaned --- ", caption_clean
		
	
		#convert commetn text in a matrix similarity
		matrix_comment = comment_to_matrix(comment_clean)
		matrix_comm=np.array(matrix_comment)
		print "matrix comment dim", matrix_comm.shape
		#print DataFrame(matrix_comm)
		#print "------------------------------------------end comment------------------------------" 
		matrix_caption = caption_to_matrix(caption_clean, comment_clean)
		#precompute squared norms of data points
		
		matrix_cap= np.array(matrix_caption)
		print "matrix caption", matrix_cap.shape
		#print DataFrame(matrix_cap)

		#print "------------------------------------------end caption------------------------------" 
	
		#print and plot clusters
		nclusters= 5
		clusters, labels, centers, predicted_label = cluster_comments(matrix_comment, nclusters, matrix_caption)
		print "matrix centroid ", centers.shape
		#print DataFrame(centers)
		#print "------------------------------------------end centroid------------------------------"
		labels_cap, mindist_caption = pairwise_distances_argmin_min(X=matrix_caption, Y=centers, metric='euclidean', metric_kwargs={'squared': False})
		#print "mindist_caption"
		#print mindist_caption
		dist_caption = pairwise_distances(X=matrix_caption, Y=centers, metric='euclidean')
		#print "dist_caption"
		#print dist_caption
		print "------------------------------------------caption - centroid------------------------------"
		labels_comm, mindist_comment = pairwise_distances_argmin_min(X=matrix_comment, Y=centers, metric='euclidean', metric_kwargs={'squared': True})
		#print "mindist_comment"
		#print mindist_comment
		dist_comment = pairwise_distances(X=matrix_comment, Y=centers, metric='euclidean')
		#print "dist_comment"
		#print dist_comment
		print "------------------------------------------comment - centroid------------------------------"


		caption_list = caption_clean.split()
	
		for cluster in range(nclusters):
			print "cluster ",cluster,":"
			for j, caption in enumerate(caption_list):
				predicted_cluster = predicted_label[j] 
				if (cluster==predicted_cluster):
					print "caption word:", caption," - cluster:", predicted_label[j] 
				
			for i,sent in enumerate(clusters[cluster]):
				text=None
				counter=0
				
				for word in comment_clean.split():
					text=word
					if (counter==sent):
						print "comment word",sent,": ", text
					counter+=1
	
		#closest, _ = pairwise_distances_argmin_min(centers, matrix_caption)
		#print closest

		#word near to the centroid
		#order_centroids=centers.argsort()[:, ::-1]
		#print order_centroids
		
		#for cluster in range(nclusters):
			#print "cluster ",cluster,":"
			#5 most common word
		#	for ind in order_centroids[cluster, :8]: 
		#		text=None
		#		counter=0
				#print "ind", ind
		#		for word in comment_clean.split():
		#			text=word
		#			if (counter==ind):
		#				print "comment word",ind,": ", text
		#			counter+=1
					
				#rint(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
		plot_cluster(matrix_comment, labels,predicted_label, centers, img_id, matrix_caption, caption_clean)
		#print "--------------------------------------------------
		
