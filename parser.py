import json
import spacy

from spacy.lang.en import English
nlp = English()

caption = json.load(open('caption.json'))
comment = json.load(open('comments.json'))


for caption_text in caption['results']:
	doc_caption = nlp((caption_text['caption']))
	print (doc_caption, " " +"\n")
	for token_caption in doc_caption:
		if (token_caption.is_stop==False):
			print (token_caption, " --> ", token_caption.lemma_)
  
  
for comment_text in comment['gresults']:
	for text in comment_text['comments']:
		doc_comment = nlp((text['text']))
		print (doc_comment, " " +"\n")
		for token_comment in doc_comment:
			if (token_comment.is_stop==False):
				print (token_comment, " --> ", token_comment.lemma_)
