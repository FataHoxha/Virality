import json
import spacy

from spacy.lang.en import English
nlp = English()
json_comment_name='100612175927429294541.json'
caption = json.load(open('caption.json'))
comment = json.load(open(json_comment_name))
obj={}
data=[]
counter_total_elemt=0
with open('output/comments_out.json','w') as jfile:
	#jfile.write('{"gresults":[')
	for img_caption in caption['results']:
		counter_total_elemt+=1
		for img_comment in comment['gresults']:
		
			#compare if the images have the same id
			if(img_caption['image_id']==img_comment['id']):
			
				#print("-------------caption lemmatization------------")
				doc_caption = nlp(img_caption['caption'])
			
				for token_caption in doc_caption:
					counter_total_word=0
					text_comment_single= None
					total_comment=0
				
					if (token_caption.is_stop==False):
						token_caption=token_caption.lemma_
					text_comm={}
					all_comm={}
					for text_comment in img_comment['comments']:
					
						#use lemmatization to "normalize the text"
						#print("-------------comment lemmatization------------")
						curr_doc_comment=None
						text_comment_id=None
						
						counter_single_comment = 0
						#prev_doc_comment=None
						doc_comment = nlp(text_comment['text'])
						text_comment_id= text_comment['id']
						all_comm[text_comment['id']] = (text_comment['text'])
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
									string_token_comment=token_comment
									counter_total_word+=1
									counter_single_comment+=1
									text_comment_single = text_comment['text']
									doc_comment=str(doc_comment)
									curr_doc_comment=doc_comment
									text_id=text_comment_id
									#add a counter for the duplicate word inside the same text
									# if curr_doc_comment == prev_doc_comment:
										#print  counter_single_doc ++
							
						if (counter_total_word > 0) and (curr_doc_comment!=None) :
							
							text_comm[text_comment['id']] = (curr_doc_comment, counter_single_comment)
							print "String: ", string_token_comment,
							print "COMMENT ID", 
							print "CURRENT DOC COMMENT", curr_doc_comment
							print "occurence per signle comment:", counter_single_comment
							if (text_id!=None):		
								#print "occurence per signle comment:", counter_single_comment
								total_comment+=1
								
							
					
					if (counter_total_word > 0) and (text_comment_single!=None) :
						img_id=img_comment['id']
						
						obj={"id":img_comment['id'],
							 "plusoners": img_comment['plusoners'] ,
							 "reshare": img_comment['reshares'],
							 "replies" : img_comment['replies'],
							 "word_match": string_token_comment, 
							 "total_occurence": counter_total_word,
							 "different_comment":total_comment,
							 "caption":img_caption['caption'],
							 "comments":[{'text_id':key,"text":value1,"counter":value2} for key,(value1,value2) in text_comm.items()],
							 "all_comments":[{'text_id':key,"text":value} for key,value in all_comm.items()]}
						
						data.append(obj)
	final_obj={"results":data, "total_element": counter_total_elemt}
	json.dump(final_obj, jfile, indent=4)
	print "done!"	
						#print "---------------------------------------------------------------"
						#print "String: ", string_token_comment, "--> Total Occurrence: ", counter_total_word , "--> in different comment", total_comment
						#print "Caption: ", img_caption['caption']
						#print "Image id: ", img_comment['id'], "--> in different comment", total_comment
						#print "Like: ", img_comment['plusoners'] , " - Reshare: ", img_comment['reshares'], " - Comment: " , img_comment['replies']
		
