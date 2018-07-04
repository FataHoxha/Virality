import json
import spacy

from spacy.lang.en import English
nlp = English()
json_comment_name='input_json/100243859631724702467.json'
caption = json.load(open('input_json/cap_100243859631724702467.json'))
comment = json.load(open(json_comment_name))
obj={}
data=[]
counter_total_elemt=0
with open('output/comm_cap_100243859631724702467.json','w') as jfile:
	#jfile.write('{"gresults":[')
	for img_caption in caption['results']:
		for img_comment in comment['gresults']:
			#compare if the images have the same id
			if(img_caption['image_id']==img_comment['id']):
				all_comm={}
				for text_comment in img_comment['comments']:
					text_comment_id= text_comment['id']
					all_comm[text_comment['id']] = (text_comment['text'])
						
				obj={"id":img_comment['id'],
					 "plusoners": img_comment['plusoners'] ,
					 "reshare": img_comment['reshares'],
					 "replies" : img_comment['replies'],
					 "caption":img_caption['caption'],
					 "all_comments":[{'text_id':key,"text":value} for key,value in all_comm.items()]}
						
				data.append(obj)
	final_obj={"results":data, "total_element": counter_total_elemt}
	json.dump(final_obj, jfile, indent=4)
	print "done!"	
						
