import json
import spacy
import collections
import re
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
nlp = English()

#comment = json.load(open('comments_a.json'))
 #Added extra word to the set of stops word

#Function to parse the html code -> delete everithing in the middle of <>
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
jfile='100612175927429294541.json'
#import json and the comment part
with open(jfile, 'r') as file:
	comment = json.load(file)
	for img_comment in comment['gresults']:
		print img_comment['id']
		for text_comment in img_comment['comments']:
			text = text_comment['text']
		
			#remove the html code
			text=cleanhtml(text)
		
			#replace the broken text
			text.replace('&lt;smile&gt;'," ")
			text = re.sub(r'&#39;', " ", text)
			text = re.sub(r'&quot;', " ", text)
			text = re.sub(r'&amp', " ", text)
			text = re.sub(r'\u3000'," ", text)
			text = re.sub(r'\u00a0', " ", text)
			#principal contracted form in english
			text = re.sub(r'don t', "do not", text)
			text = re.sub(r'aren t', "are not", text)
			text = re.sub(r'isn t', "is not", text)
			text = re.sub(r'can t', "can not", text)
			text = re.sub(r'wouldn t', "would not", text)
			text = re.sub(r'doesn t', "does not", text)
			text = re.sub(r'wasn t',"was not", text)
			text = re.sub(r'aren t', "are not", text)
			text = re.sub(r'shouldn t t', "should not", text)
			text = re.sub(r'hasn t', "has not", text)
			text = re.sub(r'didn t', "did not", text)
			text = re.sub(r'couldn t', "could not", text)
			text = re.sub(r' s ', " is ", text)
			text = re.sub(r' m ', " am ", text)
			
		
			#remove the punctation
			text = re.sub(r'[^\w\s]',' ',text)
			
			text = str(text).replace('\t', ' ').replace('\n', '');
			#put everything in lowercase
			text = text.lower()
			#remove the double space
			text = text.strip();
			text = re.sub(' +', " ", text);
			text_comment['text']= text
		print "----------------------------------------------------------------------"
with open(jfile, 'w') as file:
    json.dump(comment, file, indent=4)

                
