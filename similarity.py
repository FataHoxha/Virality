import json
import spacy
import collections
import numpy as np
import re
import statistics
from matplotlib import cbook
import matplotlib.pyplot as plt
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
jfile = json.load(open('output/comm_cap_100300281975626912157.json'))
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

def sentence_to_matrix(text1,text2):
	doc1 = nlp(text1)
	doc2 = nlp(text2)
	similarity = doc1.similarity(doc2)
	return similarity

def cluster_comments(matrix_comment, nb_of_clusters, matrix_caption):
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
	prediction=kmeans.predict(matrix_caption)

	return dict(clusters),labels, centers, prediction
	
	

def plot_cluster(matrix_comment,labels,predicted_label, centers,img_id, matrix_caption, caption, comment):
	cdict = {0: 'red', 1: 'blue', 2: 'green', 3:'dodgerblue', 4:'purple',5: 'orchid', 6: 'skyblue', 7: 'pink', 8: 'yellow', 9: 'turquoise', 10: 'violet',11:'orange',12: 'royalblue', 
				13:'mediumspringgreen', 14:'aqua', 15:'firebrick', 16:'silver', 17:'gold', 18:'bisque', 19:'black', 20:'navy', 21:'teal', 22:'blueviolet', 23:'brown', 24:'burlywood',
				25:'cadetblue', 26:'chartreuse', 27:'chocolate', 28:'coral', 29:'cornflowerblue', 30:'tomato', 31:'crimson', 32:'cyan', 33:'darkblue', 34:'darkcyan',35:'darkgoldenrod',
				36:'darkgray', 37:'darkgreen', 38:'darkkhaki', 39:'darkmagenta', 40:'darkolivegreen', 41:'darkorange', 42:'darkorchid', 43:'darkred', 44:'darksalmon', 45:'darkseagreen',
				46:'darkslateblue', 47:'darkslategray', 48:'darkturquoise', 49:'darkviolet', 50:'deeppink', 51:'deepskyblue', 52:'dimgray', 53:'greenyellow'}
	"""cdict= {0:'aliceblue',1:'antiquewhite',2:'aqua',3:'aquamarine',4:'azure',5:'beige',6:'bisque',7:'black',8:'blanchedalmond',9:'blue',10:'blueviolet',11:'brown',12:'burlywood',
			13:'cadetblue',	14:'chartreuse',15:'chocolate',16:'coral',17:'cornflowerblue',18:'cornsilk',19:'crimson',20:'cyan',21:'darkblue',22:'darkcyan',23:'darkgoldenrod',24:'darkgray',
			25:'darkgreen',26:'darkkhaki',27:'darkmagenta',28:'darkolivegreen',29:'darkorange',30:'darkorchid',31:'darkred',32:'darksalmon',33:'darkseagreen',34:'darkslateblue',
			35:'darkslategray',36:'darkturquoise',37:'darkviolet',38:'deeppink',39:'deepskyblue',40:'dimgray',41:'dodgerblue',42:'firebrick',43:'floralwhite',44:'forestgreen',
			45:'fuchsia',46:'gainsboro',47:'ghostwhite',48:'gold',49:'goldenrod',50:'gray',51:'green',52:'greenyellow',53:'honeydew',54:'hotpink',55:'indianred',56:'indigo',57:'ivory',
			58:'khaki',59:'lavender',60:'lavenderblush',61:'lawngreen',62:'lemonchiffon',63:'lightblue',64:'lightcoral',65:'lightcyan',66:'lightgoldenrodyellow',67:'lightgreen',68:'lightgray',
			69:'lightpink',70:'lightsalmon',71:'lightseagreen',72:'lightskyblue',73:'lightslategray',74:'lightsteelblue',75:'lightyellow',76:'lime',77:'limegreen',78:'linen',79:'magenta',
			80:'maroon',81:'mediumaquamarine',82:'mediumblue',83:'mediumorchid',84:'mediumpurple',85:'mediumseagreen',86:'mediumslateblue',87:'mediumspringgreen',88:'mediumturquoise',
			89:'mediumvioletred',90:'midnightblue',91:'mintcream',92:'mistyrose',93:'moccasin',94:'navajowhite',95:'navy',96:'oldlace',97:'olive',98:'olivedrab',99:'orange',100:'orangered',
			101:'orchid',102:'palegoldenrod',103:'palegreen',104:'paleturquoise',105:'palevioletred',106:'papayawhip',107:'peachpuff',108:'peru',109:'pink',110:'plum',111:'powderblue',
			112:'purple',113:'red',114:'rosybrown',115:'royalblue',116:'saddlebrown',117:'salmon',118:'sandybrown',119:'seagreen',120:'seashell',121:'sienna',122:'silver',123:'skyblue',
			124:'slateblue',125:'slategray',126:'snow',127:'springgreen',128:'steelblue',129:'tan',130:'teal',131:'thistle',132:'tomato',133:'turquoise',134:'violet',135:'wheat',136:'white',
			137:'whitesmoke',138:'yellow',139:'yellowgreen'}
	"""
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
	#pca_model = PCA(n_components=2)
	
	tsne_model = TSNE(n_components=2, perplexity=30.0, learning_rate=400.0, metric='euclidean', random_state=0)
	#mds_model = MDS(n_components=2, metric='euclidean', random_state=1)
	dense_mat= plot_matrix.toarray()
	plot_matrix_embedded =tsne_model.fit_transform(dense_mat)
	#plot_matrix_embedded=pca_model.fit_transform(dense_mat)

	
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
	

	#plot COMMENT 
	fig, ax = plt.subplots()
	counter_cent = 0
	for l in np.unique(labels):
		dist_comm_cent=[]
		ix = np.where(labels == l)
		ax.scatter(scatter_comm_x[ix], scatter_comm_y[ix], c = cdict[l], label = l, s = 30)

		x_comm=scatter_comm_x[ix]	
		y_comm=scatter_comm_y[ix]
		c_x=np.mean(x_comm)
		c_y=np.mean(y_comm)
		print "c_x: ", c_x, "c_y: ", c_y
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
		ax.scatter(c_x,c_y, color='black', s=60, alpha=0.7)
		circle = Circle((c_x, c_y), median, color = cdict[l], alpha=0.3)
		ax.add_artist(circle)
		ax.text(c_x,c_y,counter_cent, size=12)
		counter_cent+=1
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title="Cluster color")
	
	#plot text comment
	#plot CAPTION with text
	comment_list = comment.split()
	for i, comment in enumerate(comment_list):
		plt.text(scatter_comm_x[i],scatter_comm_y[i], comment_list[i], size=8)
	
	
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
		plt.scatter(scatter_cap_x[i],scatter_cap_y[i],  s=110, color=cdict[c], marker='+')
		plt.text(scatter_cap_x[i],scatter_cap_y[i], caption_list[i], size=14)
	
	plt.savefig('median_tsne_'+str(img_id) + '.pdf', bbox_inches='tight')
	plt.show()

if __name__ == "__main__":
	
	#get caption
	#clean caption
	for post in jfile['results']:
		img_id=post['id']
		print ""
		print img_id
		caption = get_caption(img_id)
		comment = get_comment(img_id)
		
		#print "---------------------------------- COMMENT----------------------------------" 
		#print "Comment ---> "
		#print comment
		comment_clean=clean_text(comment)
		
		#print "Comment cleaned ---> "
		#print comment_clean
		#print ""
		#print "---------------------------------- CAPTION----------------------------------" 
		#print "Caption ---> "
		print caption, "\n"
		caption_clean=clean_text(caption)
		print "Caption cleaned ---> ", caption_clean, "\n"
	
		if (len(comment_clean.split())>=15):
			
			#convert commetn text in a matrix similarity
			matrix_comment = comment_to_matrix(comment_clean)
			matrix_comm=np.array(matrix_comment)
			print "matrix comment dim", matrix_comm.shape
		
		
		
			#print "------------------------------------------end comment------------------------------" 
		
			matrix_caption = caption_to_matrix(caption_clean, comment_clean)
			#precompute squared norms of data points
		
			matrix_cap= np.array(matrix_caption)
			#print "matrix caption", matrix_cap.shape
			#print DataFrame(matrix_cap)

			#print "------------------------------------------end caption------------------------------" 
	
			#print and plot clusters 
			#k= 15% of the length of the length of the document comments"""
			nclusters=0
			comment_length= len(comment_clean.split())
			#print "comment length", comment_length	
			if comment_length> 400:
				nclusters= int((5 * comment_length) / 100.0)
			else:
				nclusters= int((10 * comment_length) / 100.0)
			#print "nclusters", nclusters
			if nclusters >= 2:
				clusters, labels, centers, predicted_label = cluster_comments(matrix_comment, nclusters, matrix_caption)

				labels_cap, mindist_caption = pairwise_distances_argmin_min(X=matrix_caption, Y=centers, metric='euclidean', metric_kwargs={'squared': False})
				#print "mindist_caption"
				#print mindist_caption
				dist_caption = pairwise_distances(X=matrix_caption, Y=centers, metric='euclidean')
			
				#print "------------------------------------------caption - centroid------------------------------"
				labels_comm, mindist_comment = pairwise_distances_argmin_min(X=matrix_comment, Y=centers, metric='euclidean', metric_kwargs={'squared': True})
				#print "mindist_comment"
				#print mindist_comment
				dist_comment = pairwise_distances(X=matrix_comment, Y=centers, metric='euclidean')
				"""
				print "euclideian distance comment-centroid"
				print dist_comment
				#print "---------------------------------comment - centroid------------------------------"
				"""

				caption_list = caption_clean.split()
				#list median distance comment - centroid
				median_dist = []
				for cluster in range(nclusters):
					cap_sim=[]
					print "\n"
					print "cluster ",cluster,":"
					for j, caption in enumerate(caption_list):
				
						predicted_cluster = predicted_label[j] 
						if (cluster==predicted_cluster):
							#print "caption:", caption," - cluster:", predicted_label[j] , " - distance: ", min(dist_caption[j])
							print "CAPTION:", caption #," - cluster:", predicted_label[j] , " - distance: ", min(dist_caption[j])
							cap_sim.append(caption)
					dist = []
					dist_intern=0
					count=0
					comm_sim=[]
				
					for i,sent in enumerate(clusters[cluster]):
						counter=0
						text=None
						for word in comment_clean.split():
							text=word
						
							if (counter==sent):
								#print "comment: ",sent," : ", text, " - distance: ", dist_comment[sent][cluster]
								print "comment: ", text
								#print matrix_comm[sent]
								dist_intern=dist_comment[sent][cluster]
								dist.append(dist_intern)

								#if (text not in (comm_sim)):
								comm_sim.append(text)
								
						
								count+=1
							prev_text=text
							counter+=1
				
					#print cap_sim
					s=' '.join(comm_sim)
					if cap_sim:
						for i in cap_sim:
							comm_sim_matrix= caption_to_matrix(i,s)
							comm_sim_matrix=np.array(comm_sim_matrix)
							print i,"\t", np.mean(comm_sim_matrix)
							print comm_sim_matrix
							#print "mean", np.mean(comm_sim_matrix)
							#print " "
					#print "dist totale:", (dist), count
					#avg = (dist/count)
					median=statistics.median(dist)
					median_dist.append(median)
					#print "median-->", median
					#print ""
				#print "median dist", median_dist
				#print "euclideian distance caption-centroid"
				#print dist_caption
				
				for i, caption in enumerate(dist_caption):
					#print "caption", i ,caption
					#print "median dist"
					#print median_dist
					index=0
					for cap, m_dist in zip(caption, median_dist):
					
						#print "cap",cap , "m_dist", m_dist
						if (cap<=m_dist):
							print "caption",i, "inside the median of cluster:",index, "caption:", cap,"median:", m_dist
					
						index+=1
						
				closest, _ = pairwise_distances_argmin_min(centers, matrix_caption)

				"""
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
								print "comment word: ",ind,": ", text, " - distance: ", dist_comment[ind][cluster]	
							counter+=1
				"""
		
				plot_cluster(matrix_comment, labels,predicted_label, centers, img_id, matrix_caption, caption_clean, comment_clean) 
			else:
				print "number of cluster <2, not enough words in the comment"
			#print "--------------------------------------------------
		
		else:
			print "not enught word in comment"
		
