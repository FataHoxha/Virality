import sys
from gensim.models import word2vec
import gensim.models.keyedvectors as word2vec
def w2v(s1,s2,wordmodel):
	if s1==s2:
		return 1.0
	s1words=s1.split()
	s2words=s2.split()
	s1wordsset=set(s1words)
	s2wordsset=set(s2words)
	vocab = wordmodel.vocab #the vocabulary considered in the word embeddings
	if len(s1wordsset & s2wordsset)==0:
		return 0.0
	for word in s1wordsset.copy(): #remove sentence words not found in the vocab
		if (word not in vocab):
			s1words.remove(word)
	for word in s2wordsset.copy(): #idem
		if (word not in vocab):
			s2words.remove(word)
	return wordmodel.n_similarity(s1words, s2words)

if __name__ == '__main__':
	wordmodelfile="/home/fhoxha/Documents/neuraltalk/model/GoogleNews-vectors-negative300.bin.gz"
	wordmodel= word2vec.KeyedVectors.load_word2vec_format(wordmodelfile, binary=True)
	#embed_map = word2vec..load_word2vec_format(path_to_word2vec, binary=True)
	s1="woman ride bike street"
	s2="stunning model"
	print "sim(s1,s2) = ", w2v(s1,s2,wordmodel)
	s3="woman ride bike street"
	s4=" like the bicycle"
	print "sim(s3,s4) = ", w2v(s3,s4,wordmodel)
	s5="woman ride bike street"
	s6="watched the Temple burn last night live on Ustream and was moved by the sad so longs to dearly departed that were posted to the chat"
	print "sim(s5,s6) = ", w2v(s5,s6,wordmodel)
	

	

