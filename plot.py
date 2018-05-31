import json
import math
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


results = json.load(open('output/comments.json'))
total_ids = []
counter_id =0
comments = {}
for result in results['results']:
	
	#total images involved without duplicates
	total_ids.append(result['id'])
	counter_id+=1
	#comment involved
	
	comments[counter_id]=(result['id'],result['word_match'],result['different_comment'])

print (comments)
