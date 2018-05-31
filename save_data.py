import json



def getImageId(image_id):
	
	print image_id
	return image_id	
	

file = open("filename", "w")

output = {"product1": image_id}
json.dump(output, file)
file.close()

