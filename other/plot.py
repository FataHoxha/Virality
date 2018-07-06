import json
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


results = json.load(open('output/comments.json'))
total_ids = []
data_dict = defaultdict(list)

for result in results['results']:
	
	#total images involved without duplicates
	total_ids.append(result['id'])

	#comment involved
	
	data_dict[result['id']].append(result['different_comment'], result['replies'], result['word_match'])

print (total_ids)
print "cose",(data_dict.items())
