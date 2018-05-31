import json
import spacy
import collections
nlp = spacy.load('en_core_web_sm')

jfile = json.load(open('output/comments_out.json'))

comment_doc=collections.defaultdict(dict)
'''
#foreach id in json put toghether all the comments -> to have a doc
for img_id in jfile['results']:
	text_complete=" "
	text_comment=[]
	
	for comment in img_id['all_comments']:
		doc_comment = nlp(comment['text'])
		string_text =' '
		#print "DOC COMMENT --->", doc_comment
		for token_comment in doc_comment:
			token_comment = token_comment.lemma_
			doc_token = nlp(token_comment)
			for token in doc_token:
				if (token.is_stop==False):
					if (str(token) != '-PRON-'):
						token_comm=str(token)
					#print token_comm
					string_text+=' '+token_comm
	#	print string_text
		text_complete+=string_text
	#print text_complete
	comment_doc[img_id['id']]=(text_complete)
	'''
'''
#take each text as doc and tokenize
for id, text in comment_doc.iteritems():
	
	
	if (id=="102865263115020893218.5811636111390168418"):
		print id
		print text'''
def word_tokenizer(text):
	token_result =[]
	tokens = nlp(unicode(text))
	#tokens = nlp(u'dog cat banana')
	print len(tokens)
	for token1 in tokens:	
		if len(token1)>1:
			for token2 in tokens:
				if len(token2)>1:
					token11=token1.text
					token22=token2.text
					similarity=token1.similarity(token2)
					#print  token1.text, token2.text, token1.similarity(token2)
					#similarity=token1.similarity(token2)
					res =(token11,token22, similarity)
					token_result.append(res)
	return token_result

'''
def cluster_sentences(sentences, nb_of_clusters=5):

	#the values comes from the similarity function -> need to construct a similarity matrix
	tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
										stop_words=stopwords.words('english'),
										max_df=0.9,
										min_df=0.1,
										lowercase=True)
										
            #builds a tf-idf matrix for the sentences
	tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
	kmeans = KMeans(n_clusters=nb_of_clusters)
	kmeans.fit(tfidf_matrix)
	clusters = collections.defaultdict(list)
	for i, label in enumerate(kmeans.labels_):
z clusters[label].append(i)
            return dict(clusters)'''


if __name__ == "__main__":
	sentence = ("love nice picture nice picture care hand great eye great idea idea hard believe believe instantly focus hand look background blend hand order add bead water perspective strand water suppose hand idea great")
	
	list_1 = word_tokenizer(sentence)
	#print list_1
	
    # '''nclusters= 3
        #    clusters = cluster_sentences(sentences, nclusters)
       #   for cluster in range(nclusters):
        #            print "cluster ",cluster,":"
        #            for i,sentence in enumerate(clusters[cluster]):
        #                    print "\tsentence ",i,": ",sentences[sentence]'''
                            
                            
                            
                            
                            
                            
                            
