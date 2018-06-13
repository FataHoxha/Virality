
		Image Virality

--------------------------------------------------------------------------------

Comments-Caption Analysis
Input -> comments.json, caption.json

--------------------------------------------------------------------------------
- Clean: remove number, html code, punctation, capslock.
	- clean_data.py -> input: input_json/comments.json


- Compare: foreach post compare comments and caption to find word match.
	- compare.py ->  input: input_json/comments.json
				 -> output: output/comment-caption.json
				
				 
- Find Similarity: foreach post cluster similar word(k-means), predict caption,
				   dim reduction t-SNE and plot the results.
	- similarity.py ->  input: output/comments.json

--------------------------------------------------------------------------------
