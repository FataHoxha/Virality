import json
import spacy
import collections
from spacy.lang.en import English
nlp = English()
from spacy.lang.en.stop_words import STOP_WORDS
json_comment_name='comments_a.json'
caption = json.load(open('caption.json'))
comment = json.load(open(json_comment_name))

comment_dict=collections.defaultdict(dict)
image_id=[]
comment_token=collections.defaultdict(dict)

#print (STOP_WORDS)
text=None
for img_caption in caption['results']:
	
	for img_comment in comment['gresults']:
					#compare if the images have the same id
		if(img_caption['image_id']==img_comment['id']):
			
			#print("-------------caption lemmatization------------")
			doc_caption = nlp(img_caption['caption'])
			
			for token_caption in doc_caption:
				
			
				if (token_caption.is_stop==False):
					token_caption=token_caption.lemma_
					
				for text_comment in img_comment['comments']:
					
					#use lemmatization to "normalize the text"
					#print("-------------comment lemmatization------------")

					doc_comment = nlp(text_comment['text'])

					#print "DOC COMMENT --->", doc_comment
					for token_comment in doc_comment:
							
						if (token_comment.is_stop==False):
							token_comment = token_comment.lemma_
							
							#convert from unicode to string 
							token_comment=str(token_comment)
							token_caption=str(token_caption)
							
							#find the same string and count the occurence of that string 
							#in all the comment related with that id
							if token_caption==token_comment:
								if img_comment['id'] not in image_id:
									image_id.append(img_comment['id'])
								
		for img_id in image_id:
			
			if (img_id==img_comment['id']):
				#print img_id 
				for text_comment in img_comment['comments']:
					comment_dict[img_comment['id']][text_comment['id']]=(text_comment['text'])
					#print img_comment['id']
					#print text_comment['id']
					#print text_comment['text']

#print comment_dict
for key, value in comment_dict.iteritems():
	print "img_id", key
	#token=None
	comment_list=[]
	token_list=[]
	for key1, value1 in value.iteritems():
		#print "comment_id", key1
		#unicode(value1)
		doc_value1=nlp(value1)
		
		for token_value1 in doc_value1 :
			token_value1=token_value1.lemma_
			token_value1=nlp(token_value1)
			for token_val1 in token_value1:
				
				#if ((token_val1).is_stop==False):
					
						#print token_value1
						#
				counter_comment_word=0
				for key2, value2 in value.iteritems():
					if (key1!=key2):
						#print key1, key2							
						doc_value2=nlp(value2)
						for token_value2 in doc_value2 :
							token_value2=token_value2.lemma_
							token_value2=nlp(token_value2)
							for token_val2 in token_value2:
								print(token_val1.similarity(token_val2))
								if ((token_val2).is_stop==False):
									print(token_val1.similarity(token_val2))
									#convert from unicode to string 
									token_val1=str(token_val1)
									token_val2=str(token_val2)
							
										#find the same string and count the occurence of that string 
										#in all the comment related with that id
									if token_val1==token_val2:
											#print token_val1
											#print token_val1
										if token_val1 not in token_list:
												
											token_list.append(token_val1)
												#val=(token_list,  counter_comment_word)
												#token_list.append(val)
											counter_comment_word+=1
			
	print token_list
											#if token_value1 not in token_list:
											#	token_list.append(token_value1)
											#	print token_list
											#	comment_token[key][key1]=(token_list)
		
			#print "text id2", key1
			#print "text comment2", value1
#for key3, value3 in comment_token.iteritems():
#	print key3
#	for key31, value31 in value3.iteritems():
#		print key31		
#		print value31
	print "-----------------------------------------"						







