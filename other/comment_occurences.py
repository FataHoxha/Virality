import json
import math
import matplotlib.pyplot as plt
import numpy as np

def truncate(number, digits):
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper

results = json.load(open('output/comments.json'))
ids = [] #without duplicates
total_ids = [] #with duplicates
comments={}
counter_id=0
print "--- Word with occurence in comment: "
for result in results['results']:
	print result['word_match']
	#total id, with duplicates
	total_ids.append(result['id'])
	#total images involved without duplicates
	if result['id'] not in ids:
		ids.append(result['id'])
	#comment involved
	counter_id+=1
	comments[counter_id]=(result['id'],result['different_comment'], result['replies'], result['word_match'])

print len(total_ids)
print "--- Number of images that have a match: ", len(ids)

#element with match/total element in origianl dataset

ids=len(ids) #without duplicates
total_ids=len(total_ids)
total_element = results['total_element'] #number of post analyzed during the match of the word

res = (float(ids)/float (total_element))*100
res = truncate(res,2)
print "% of images that have a match in their comment: ", res

# comment involved
total=[]
for key,(id,different_comment,total_comment, word) in comments.items():
	#Avarage of total comment involved for each images
	#different comment: number of comment that have at least one match
	#total comment: number of the comment for each post
 	result= (float(different_comment)/float(total_comment))*100
 	result = truncate(result,2)
	total.append(result)
print "For each image analyzed:",ids ," % of comment with match", total

result1=(sum(total))/total_ids
result1=truncate(result1,2)
#Avarage of total comment involved for all images
print "% of comment with a match for alla images" , result1


#Plot the graph with the total number of comment (for each post) and 
#the number of comment wich have an occurence of a specific word
n_groups=total_ids

data1=[]
data2=[]
data3=[]

for key,(id,different_comment,total_comment, word) in comments.items():
	data1.append(total_comment)
	data2.append(different_comment)
	data3.append(word)

#plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, data1, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Total comment')
 
rects2 = plt.bar(index + bar_width, data2, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Comment with occurence')
 
plt.xlabel('comments per image')
plt.ylabel('occurencce')
plt.xticks(index + bar_width, (data3))
plt.legend()
 
plt.tight_layout()
plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
