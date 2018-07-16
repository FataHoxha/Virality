import json
import spacy
import collections
import numpy as np
import re
import statistics
from matplotlib import cbook
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib as mpl
from matplotlib.patches import Circle
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import  pairwise 
from scipy.spatial import distance
from scipy.sparse import vstack
nlp = spacy.load('en_core_web_lg')
jfile = json.load(open('output/comm_cap_100091697767366769792.json'))
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
				if (len(token_comm)>2):
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


def cluster_comments(matrix_comment, nb_of_clusters):
    #create k means object and pass to it the number of clusters
	kmeans = KMeans(nb_of_clusters,random_state=None)
	#pass the matrix to the fit method of kmeans to compute k-means clustering.
	kmeans.fit(matrix_comment)
	#y_kmeans = kmeans.predict(matrix_comment)
	#print "labels:", 
	kmeans.labels_
	clusters = collections.defaultdict(list)
	for i, label in enumerate(kmeans.labels_):
		clusters[label].append(i)
	
	centers = kmeans.cluster_centers_
	labels=kmeans.labels_	
	

	return dict(clusters),labels, centers

def plot_cluster_TSNE(matrix_comment,labels,predicted_label, centers,img_id, matrix_caption, caption, comment):
	cdict = {0: 'red', 1: 'blue', 2: 'green', 3:'dodgerblue', 4:'purple',5: 'orchid', 6: 'skyblue', 7: 'pink', 8: 'yellow', 9: 'turquoise', 10: 'violet',11:'orange',12: 'royalblue', 
				13:'mediumspringgreen', 14:'aqua', 15:'firebrick', 16:'silver', 17:'gold', 18:'bisque', 19:'black', 20:'navy', 21:'teal', 22:'blueviolet', 23:'brown', 24:'burlywood',
				25:'cadetblue', 26:'chartreuse', 27:'chocolate', 28:'coral', 29:'cornflowerblue', 30:'tomato', 31:'crimson', 32:'cyan', 33:'darkblue', 34:'darkcyan',35:'darkgoldenrod',
				36:'darkgray', 37:'darkgreen', 38:'darkkhaki', 39:'darkmagenta', 40:'darkolivegreen', 41:'darkorange', 42:'darkorchid', 43:'darkred', 44:'darksalmon', 45:'darkseagreen',
				46:'darkslateblue', 47:'darkslategray', 48:'darkturquoise', 49:'darkviolet', 50:'deeppink', 51:'deepskyblue', 52:'dimgray', 53:'greenyellow'}

	"""
	-->1) Prepare the matrix to be plotted = comment+caption+cluster
	
		append comment-caption-centers in same matrix -> than pass it do dim reduction
		save num column&row -> use them as range to plot matrix content
	"""
	plot_matrix = []
	#matrix comment converted to np array, col&row saved
	matrix_comment=np.array(matrix_comment)
	comm_num_rows, comm_num_cols = matrix_comment.shape
	
	#matrix caption converted to np array, col&row saved
	matrix_caption=np.array(matrix_caption)
	cap_num_rows, cap_num_cols = matrix_caption.shape
	
	centers=np.array(centers)
	center_num_rows, center_num_cols = centers.shape
	#stack of 3 matrix
	plot_matrix=vstack((matrix_comment,matrix_caption,centers))
	
	"""
	-->2) dim reduction of the matrix to be plotted -> 2D
	
		dim reduction on the whole matrix in a 2D as we're plotting points in a two-dimensional plane
		t-sne used to apply dim reduction
	"""
	tsne_model = TSNE(n_components=2, perplexity=30.0, learning_rate=400.0, metric='euclidean', random_state=0)
	dense_mat= plot_matrix.toarray()
	plot_matrix_embedded =tsne_model.fit_transform(dense_mat)

	"""
	-->3) Start plotting all the component of the matrix and put correct range
	"""
	#prepare scatter plot for comment
	scatter_comm_x= plot_matrix_embedded[:comm_num_rows, 0]
	scatter_comm_y= plot_matrix_embedded[:comm_num_rows, 1]
	
	#prepare scatter plot for caption
	cap_range_max=comm_num_rows+cap_num_rows
	scatter_cap_x= plot_matrix_embedded[comm_num_rows:cap_range_max, 0]
	scatter_cap_y= plot_matrix_embedded[comm_num_rows:cap_range_max, 1]
	
	#prepare scatter plot for centroids

	scatter_cent_x= plot_matrix_embedded[cap_range_max:, 0]
	scatter_cent_y= plot_matrix_embedded[cap_range_max:, 1]
	

	#--------------------------------------PRECOMPUTED CENTROID-------------------------------------------------------------------------
	#plot COMMENT 
	fig1, ax1 = plt1.subplots()
	counter_cent1 = 0
	for l in np.unique(labels):
		dist_comm_cent=[]
		ix = np.where(labels == l)
		ax1.scatter(scatter_comm_x[ix], scatter_comm_y[ix], c = cdict[l], label = l, s = 30)

		x_comm=scatter_comm_x[ix]	
		y_comm=scatter_comm_y[ix]
		c_x=np.mean(x_comm)
		c_y=np.mean(y_comm)
		for i,j in zip(x_comm, y_comm):
			x_cent=scatter_cent_x[l]
			y_cent=scatter_cent_y[l]
			
			a = np.array((i, j))
			#b = np.array((x_cent, y_cent))
			
			#calculate the new centroid
			
			b = np.array((c_x, c_y))
			dist = np.linalg.norm(a-b)
			dist_comm_cent.append(dist)
		#mean=sum(dist_comm_cent) / float(len(dist_comm_cent))
		median=statistics.median(dist_comm_cent)
		ax1.scatter(c_x,c_y, color='black', s=30, alpha=0.4)
		circle = Circle((c_x, c_y), median, color = cdict[l], alpha=0.3)
		ax1.add_artist(circle)
		ax1.text(c_x,c_y,counter_cent1, size=12)
		counter_cent1+=1
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title="TSNE Cluster color")
	
	#plot text comment
	#plot CAPTION with text
	comment_list1 = comment.split()
	for i, comment in enumerate(comment_list1):
		plt1.text(scatter_comm_x[i],scatter_comm_y[i], comment_list[i], size=8)
	
	
	#plot CENTROIDS
	#counter_cent = 0
	#for i,j in zip(scatter_cent_x,scatter_cent_y):
	
		#plt.scatter(i,j, color='black', s=100, alpha=0.7)
		#plt.text(c_x,c_y,counter_cent, size=12)
		#counter_cent+=1
	#plot CAPTION with text
	caption = caption.split()
	for i, caption in enumerate(caption_list):
		c=predicted_label[i]
		plt1.scatter(scatter_cap_x[i],scatter_cap_y[i],  s=60, color=cdict[c], marker='+')
		plt1.text(scatter_cap_x[i],scatter_cap_y[i], caption_list[i], size=14)
	
	plt1.savefig('/home/fhoxha/Documents/Virality/results/tsne_'+str(img_id) + '.pdf', bbox_inches='tight')
	#lt.savefig('/home/fhoxha/Documents/Virality/results/median_tsne_'+str(img_id) + '.pdf', format='pdf', dpi=1000, box_inches='tight')
	plt1.show()
	
if __name__ == "__main__":
	
	#get caption
	#clean caption
	for post in jfile['results']:
		img_id=post['id']
		#print ""
		#print img_id
		#caption = get_caption(img_id)
		comment = get_comment(img_id)
		comment_clean=clean_text(comment)
	print "---------------------------------- COMMENT----------------------------------" 
	print comment
	
	print "---------------------------------- COMMENT cleaned --------------------------"
	print comment_clean

	#convert commetn text in a matrix similarity
	matrix_comment = comment_to_matrix(comment_clean)
	matrix_comm=np.array(matrix_comment)
	print "matrix comment dim", matrix_comm.shape
	
	#CLUSTERING
	#k= 15% of the length of the length of the document comments
	nclusters=0
	comment_length= len(comment_clean.split())
	
	nclusters= int((5 * comment_length) / 100.0)
	
	if nclusters >= 2:
		clusters, labels, centers = cluster_comments(matrix_comment, nclusters)

		#labels_cap, mindist_caption = pairwise_distances_argmin_min(X=matrix_caption, Y=centers, metric='euclidean', metric_kwargs={'squared': False})
		#dist_caption = pairwise_distances(X=matrix_caption, Y=centers, metric='euclidean')
		for cluster in range(nclusters):
			print "\n"
			print "cluster ",cluster,":"
			
			for i,sent in enumerate(clusters[cluster]):
				counter=0
				text=None
				for word in comment_clean.split():
					text=word
					if (counter==sent):
					#print "comment: ",sent," : ", text, " - distance: ", dist_comment[sent][cluster]
						print "comment: ", text
							#print matrix_comm[sent]
					counter+=1
		closest, _ = pairwise_distances_argmin_min(centers, matrix_comment)

				
		#comment word near to the centroid
		order_centroids=centers.argsort()[:, ::-1]
		print "------------------------5word-> TOPIC OUTPUT-------------------------------" 
				
		for cluster in range(nclusters):
			print "cluster ",cluster,":"
			#5 most common word
			for ind in order_centroids[cluster, :5]: 
				text=None
				counter=0
				for word in comment_clean.split():
					text=word
					if (counter==ind):
						print "comment word: ",ind,": ", text
					counter+=1
				

