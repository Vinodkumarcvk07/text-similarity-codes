import gensim
import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans
from generate_tfidf import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import os
import sys , traceback
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from prettytable import PrettyTable
import re
import pickle
from sklearn.decomposition import SparsePCA,PCA
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity

save_preprocessed_dataset=False
create_kmeans=False
load_existing_preprocessed_files=True
open_inferring_terminal_pure_doc2vec=False
open_inferring_terminal_kmeans_doc2vec=True

dataset=[]

# Preprocessing
stopword = set(stopwords.words('english'))
ps = PorterStemmer()
TOKENS=['_DATE_','_OPERATOR_','_DISKSPACE_','_PERC_','_IP_','_PATH_','_NUM_']

def preprocess_line(line):
	line = line.lower()
	line = line.replace('\n','')

	# MultiSpace between words converted to one space
	line = re.sub(r'\s+',' ',line)

	# Date of following format converted to constant
	line = re.sub(r'\d\d/\d\d/\d\d[ _]\d\d:\d\d:\d\d',' _DATE_ ',line)
	line = re.sub(r'\d\d/\d\d/\d\d\d\d[ _]\d\d:\d\d:\d\d',' _DATE_ ',line)
	
	line = re.sub(r'\d\d:\d\d:\d\d',' _DATE_ ',line)

	line = re.sub(r'\d\d/\d\d/\d\d\d\d[ _]\d\d:\d\d',' _DATE_ ',line)
	line = re.sub(r'\d\d/\d\d[ _]\d\d:\d\d',' _DATE_ ',line)

	line = re.sub(r'\d\d/\d\d/\d\d\d\d',' _DATE_ ',line)
	line = re.sub(r'\d\d\.\d\d\.\d\d\d\d',' _DATE_ ',line)   # May conflict with IPs
	line = re.sub(r'\d\d\d\d-\d\d-\d\d',' _DATE_ ',line)

	# Percentage numbers converted to constant
	line = re.sub(r'[\d]+\.[\d]+\%',' _PERC_ ',line)
	line = re.sub(r'[\d]+\%',' _PERC_ ',line)
	# IP of machine converted to constant
	line = re.sub(r'[\d]+\.[\d]+\.[\d]+\.[\d]+',' _IP_ ',line)  
	
	# Cleaning random noise
	line = re.sub(r'[?]+',' ',line)
	line = re.sub(r'-->',' ',line)
	line = re.sub(r'->',' ',line)
	line = re.sub(r'==>',' ',line)
	line = re.sub(r'=>',' ',line)
	line = re.sub(r'= >',' ',line)
	line = re.sub(r'\*',' ',line)
	line = re.sub(r'-[-]+',' ',line)
	line = re.sub(r'\.[\.]+',' ',line)
	line = re.sub(r'\"',' ',line)
	

	# Putting Space around these characters, helps in tokenization
	line = re.sub(r'\[',' ',line)
	line = re.sub(r'\]',' ',line)
	line = re.sub(r'\;',' ',line)
	line = re.sub(r'\&',' ',line)
	line = re.sub(r'\)',' ',line)
	line = re.sub(r'\(',' ',line)

	# Replacing operations with constant
	line = re.sub(r'>=',' _OPERATOR_ ',line)
	line = re.sub(r'<=',' _OPERATOR_ ',line)
	line = re.sub(r'=',' _OPERATOR_ ',line)
	line = re.sub(r'>',' _OPERATOR_ ',line)
	line = re.sub(r'<',' _OPERATOR_ ',line)
	
	# Replacing directory path with constant 
	line = re.sub(r' [a-z]\:[\\a-z0-9\_\.]+ ',' _PATH_ ',line)
	line = re.sub(r' [a-z]\:',' _PATH_ ',line)
	#line = re.sub(r'[\\a-z0-9\_\.]+ ',' _PATH_ ',line)  # only replacing strict paths
	
	# Putting Space around these characters, helps in tokenization
	line = re.sub(r'\/',' ',line)
	line = re.sub(r'\\',' ',line)
	line = re.sub(r'\:',' ',line)
	
	# Removing space from the end and begining
	line = re.sub(r'[\s]+$','',line)
	line = re.sub(r'^[\s]+','',line)
	# Removing full stop at the end
	line = re.sub(r'[\.]$','',line)
	
	# Converting disk space to constant
	line = re.sub(r'[\d]+gb',' _DISKSPACE_ ',line)
	line = re.sub(r' [\d]+[ ]+gb ',' _DISKSPACE_ ',line)
	line = re.sub(r' [\d]+[ ]+gb$',' _DISKSPACE_ ',line)
	line = re.sub(r'[\d]+mb',' _DISKSPACE_ ',line)
	line = re.sub(r' [\d]+[ ]+mb ',' _DISKSPACE_ ',line)
	line = re.sub(r' [\d]+[ ]+mb$',' _DISKSPACE_ ',line)
	line = re.sub(r'[\d]+kb',' _DISKSPACE_ ',line)
	line = re.sub(r' [\d]+[ ]+kb ',' _DISKSPACE_ ',line)
	line = re.sub(r' [\d]+[ ]+kb$',' _DISKSPACE_ ',line)
	
	# Replacing all num with constant
	line = re.sub(r' [0-9]+ ',' _NUM_ ',line)
	line = re.sub(r' [0-9]+$',' _NUM_ ',line)
	line = re.sub(r'^[0-9]+ ',' _NUM_ ',line)
	line = re.sub(r'^[0-9]+\.[0-9]+ ',' _NUM_ ',line)  # Floating Points  / Hopefully IP addresses will not be impacted as it is already replaced with constant
	line = re.sub(r' [0-9]+\.[0-9]+ ',' _NUM_ ',line)  # Floating Points
	line = re.sub(r' [0-9]+\.[0-9]+$',' _NUM_ ',line)  # Floating Points

	# Some Random Specific Nosie Removal
	line = re.sub(r'\[[ ]+\]',' ',line)

	# MultiSpace between words converted to one space
	line = re.sub(r'\s+',' ',line)
	new_line=[]
	for wrd in word_tokenize(line):
		if wrd in TOKENS:
			#new_line.append(wrd)   # Skipping putting this as they add little value to problem
			pass
		else:
			try:
				if wrd.lower() not in stopword:
					new_line.append(ps.stem(wrd.lower()))
				else:
					continue
			except Exception as e:
				print("here {}".format(e))
	return new_line #as array

#################################################################################################
data=genfromtxt('training2.csv', delimiter =',', invalid_raise=False,dtype=None)

if load_existing_preprocessed_files == True:
	dataset = np.load('saved/preprocessed_dataset.txt.npy')
else:
	for idx,line in enumerate(data):
		new_line = preprocess_line(line[0].decode('utf-8'))
		line[0] = " ".join(new_line)
		dataset.append(line)
	dataset = np.array(dataset)

if save_preprocessed_dataset == True:
	np.save('saved/preprocessed_dataset.txt',dataset)
print("Dataset Loaded & Preprocessed {}".format(dataset.shape))
#################################################################################################

#################################################################################################
"""	 Doc2Vec Preperation  """
class LabeledLineSentence(object):
	def __init__(self, doc_list, labels_list):
		self.labels_list = labels_list
		self.doc_list = doc_list
	def __iter__(self):
		for idx, doc in enumerate(self.doc_list):
			yield gensim.models.doc2vec.LabeledSentence(doc.decode('utf-8'),[int(self.labels_list[idx])])
it = LabeledLineSentence(dataset[:,0],list(range(dataset.shape[0])))

""" Building Doc2Vec model """
if os.path.isfile('saved/doc2vec_model.bin'):
	model = gensim.models.doc2vec.Doc2Vec.load('saved/doc2vec_model.bin')
else:
	model = gensim.models.Doc2Vec(vector_size=100)
	model.build_vocab(it)
	model.train(it,total_examples=model.corpus_count,epochs=100)
	model.save('saved/doc2vec_model.bin')
#################################################################################################



#################################################################################################
""" kmeans preperation """

if create_kmeans == True:
	description_vectorizer = CountVectorizer(binary=True,decode_error='ignore')
	description = description_vectorizer.fit_transform(dataset[:,0]).todense().tolist()
	pca = PCA(n_components = 1320 )
	description_pcaed = pca.fit_transform(description)
	
	le_reportedby_int = LabelEncoder()
	ce_reportedby_int = le_reportedby_int.fit_transform(dataset[:,1]).reshape(-1,1)
	ce_reportedby = OneHotEncoder(handle_unknown='ignore')
	reportedby_onehot = ce_reportedby.fit_transform(ce_reportedby_int).todense().tolist()
	pca_reportedby = PCA ( n_components = 100 )
	reportedby_onehot_pcaed = pca_reportedby.fit_transform(reportedby_onehot)
	
	le_affectedperson_int = LabelEncoder()
	ce_affectedperson_int = le_affectedperson_int.fit_transform(dataset[:,2]).reshape(-1,1)
	ce_affectedperson = OneHotEncoder(handle_unknown='ignore')
	ce_affectedperson_onehot = ce_affectedperson.fit_transform(ce_affectedperson_int).todense().tolist()
	pca_affectedperson = PCA (n_components = 80 )
	ce_affectedperson_pcaed = pca_affectedperson.fit_transform(ce_affectedperson_onehot)
	
	le_ownergroup_int = LabelEncoder()
	ce_ownergroup_int = le_ownergroup_int.fit_transform(dataset[:,3]).reshape(-1,1)
	ce_ownergroup = OneHotEncoder(handle_unknown='ignore')
	ce_ownergroup_onehot = ce_ownergroup.fit_transform(ce_ownergroup_int).todense().tolist()    # No Need to do PCA as there are only 120 ownergroups in the available dataset
	
	le_siteid_int = LabelEncoder()
	ce_siteid_int = le_siteid_int.fit_transform(dataset[:,4]).reshape(-1,1)
	ce_siteid = OneHotEncoder(handle_unknown='ignore')
	ce_siteid_onehot = ce_siteid.fit_transform(ce_siteid_int).todense().tolist()
	
	le_assetid_int = LabelEncoder()
	ce_assetid_int = le_assetid_int.fit_transform(dataset[:,5]).reshape(-1,1)
	ce_assetid = OneHotEncoder(handle_unknown='ignore')
	ce_assetid_onehot = ce_assetid.fit_transform(ce_assetid_int).todense().tolist()
	
	le_createdby_int = LabelEncoder()
	ce_createdby_int = le_createdby_int.fit_transform(dataset[:,6]).reshape(-1,1)
	ce_createdby = OneHotEncoder(handle_unknown='ignore')
	ce_createdby_onehot = ce_createdby.fit_transform(ce_createdby_int).todense().tolist()
	
	pca_createdby = PCA(n_components = 100 )
	ce_createdby_pcaed = pca_createdby.fit_transform(ce_createdby_onehot)
	
	X_arr = []
	for idx,dsc in enumerate(description_pcaed):
		arr = []
		arr.extend(dsc)
		arr.extend(ce_createdby_pcaed[idx])
		arr.extend(reportedby_onehot_pcaed[idx])
		arr.extend(ce_affectedperson_pcaed[idx])
		arr.extend(ce_ownergroup_onehot[idx])
		arr.extend(ce_siteid_onehot[idx])
		arr.extend(ce_assetid_onehot[idx])
		X_arr.append(arr)
	
	dataset_X = np.array(X_arr)
	
	print("KMeans Training Dataset shape {}".format(dataset_X.shape))
	
	kmeans = KMeans(n_clusters = 120,random_state=0)
	
	#Training
	kmeans.fit(dataset_X)
	
	#Saving
	kmeans_clusters={}
	kmeans_clusters['cluster_dataset']={}
	kmeans_clusters['dataset_X']=dataset_X
	for i in range(120):
	        kmeans_clusters['cluster_dataset'][i]=np.where(kmeans.labels_ == i)[0]
	kmeans_clusters['kmeans']=kmeans
	pickle.dump(kmeans_clusters,open('saved/kmeans_clusters.pck','wb'))
	
	#Saving Rest
	
	pickle.dump(description_vectorizer,open("saved/description_vectorizer",'wb'))
	pickle.dump(pca,open("saved/pca",'wb'))
	
	pickle.dump(le_reportedby_int,open("saved/le_reportedby_int",'wb'))
	pickle.dump(ce_reportedby,open("saved/ce_reportedby",'wb'))
	pickle.dump(pca_reportedby,open("saved/pca_reportedby",'wb'))
	
	pickle.dump(le_affectedperson_int,open("saved/le_affectedperson_int",'wb'))
	pickle.dump(ce_affectedperson,open("saved/ce_affectedperson",'wb'))
	pickle.dump(pca_affectedperson,open("saved/pca_affectedperson",'wb'))
	
	pickle.dump(le_ownergroup_int,open("saved/le_ownergroup_int",'wb'))
	pickle.dump(ce_ownergroup,open("saved/ce_ownergroup",'wb'))
	
	pickle.dump(le_siteid_int,open("saved/le_siteid_int",'wb'))
	pickle.dump(ce_siteid,open("saved/ce_siteid",'wb'))
	
	pickle.dump(le_assetid_int,open("saved/le_assetid_int",'wb'))
	pickle.dump(ce_assetid,open("saved/ce_assetid",'wb'))
	
	pickle.dump(le_createdby_int,open("saved/le_createdby_int",'wb'))
	pickle.dump(ce_createdby,open("saved/ce_createdby",'wb'))
	pickle.dump(pca_createdby,open("saved/pca_createdby",'wb'))
else:
	# Loading
	kmeans_clusters = pickle.load(open('saved/kmeans_clusters.pck','rb'))
	dataset_X = kmeans_clusters['dataset_X']

	description_vectorizer = pickle.load(open("saved/description_vectorizer",'rb'))
	pca = pickle.load(open("saved/pca",'rb'))
	
	le_reportedby_int = pickle.load(open("saved/le_reportedby_int",'rb'))
	ce_reportedby = pickle.load(open("saved/ce_reportedby",'rb'))
	pca_reportedby = pickle.load(open("saved/pca_reportedby",'rb'))
	
	le_affectedperson_int = pickle.load(open("saved/le_affectedperson_int",'rb'))
	ce_affectedperson = pickle.load(open("saved/ce_affectedperson",'rb'))
	pca_affectedperson = pickle.load(open("saved/pca_affectedperson",'rb'))
	
	le_ownergroup_int = pickle.load(open("saved/le_ownergroup_int",'rb'))
	ce_ownergroup = pickle.load(open("saved/ce_ownergroup",'rb'))
	
	le_siteid_int = pickle.load(open("saved/le_siteid_int",'rb'))
	ce_siteid = pickle.load(open("saved/ce_siteid",'rb'))
	
	le_assetid_int = pickle.load(open("saved/le_assetid_int",'rb'))
	ce_assetid = pickle.load(open("saved/ce_assetid",'rb'))
	
	le_createdby_int = pickle.load(open("saved/le_createdby_int",'rb'))
	ce_createdby = pickle.load(open("saved/ce_createdby",'rb'))
	pca_createdby = pickle.load(open("saved/pca_createdby",'rb'))


#################################################################################################


if open_inferring_terminal_kmeans_doc2vec == True:

	dataset_X = kmeans_clusters['dataset_X']
	kmeans = kmeans_clusters['kmeans']

	#Finding Length of Vector for ownergroup,siteid,assetid
	max_ownergroup_vector_width = ce_ownergroup.transform([[ce_ownergroup.active_features_.max()]]).toarray().shape[1]
	max_siteid_vector_width = ce_siteid.transform([[ce_siteid.active_features_.max()]]).toarray().shape[1]
	max_assetid_vector_width = ce_assetid.transform([[ce_assetid.active_features_.max()]]).toarray().shape[1]
	
	input_line=""
	while True:
		try:
			t = PrettyTable(['docid','Similar Ticket Description'])
			issue_description = input("Enter Issue Description [mandatory field][press enter to submit] :")
			if issue_description.strip() == '' or issue_description.strip() == '\n':
				print("Incorrect Input!!")
				continue
			issue_createdby = input("Enter Created By [mandatory field][press enter to submit] :")
			if issue_createdby.strip() == '' or issue_createdby.strip() == '\n':
				print("Incorrect Input!!")
				continue

			issue_reportedby = input("Enter Reported By [optional][press enter to submit] :")
			if issue_reportedby.strip() == '' or issue_reportedby.strip() == '\n':
				print("will use default value")
				issue_reportedby="default"
			
			issue_affectedperson = input("Enter Affected Person [optional][press enter to submit] :")
			if issue_affectedperson.strip() == '' or issue_affectedperson.strip() == '\n':
				print("will use default value")
				issue_affectedperson="default"
			
			issue_ownergroup = input("Enter Owner Group [optional][press enter to submit] :")
			if issue_ownergroup.strip() == '' or issue_ownergroup.strip() == '\n':
				print("will use default value")
				issue_ownergroup="default"
			
			issue_siteid = input("Enter Site Id [optional][press enter to submit] :")
			if issue_siteid.strip() == '' or issue_siteid.strip() == '\n':
				print("will use default value")
				issue_siteid="default"

			issue_assetid = input("Enter Asset Id [optional][press enter to submit] :")
			if issue_assetid.strip() == '' or issue_assetid.strip() == '\n':
				print("will use default value")
				issue_assetid="default"


			issue_description = preprocess_line(issue_description)
			issue_description_ = description_vectorizer.transform([" ".join(issue_description)]).todense().tolist()
			issue_description_final = pca.transform(issue_description_)

			issue_createdby_ = le_createdby_int.transform([issue_createdby]).reshape(-1,1)
			issue_createdby__ = ce_createdby.transform(issue_createdby_).todense().tolist()
			issue_createdby_final = pca_createdby.transform(issue_createdby__)

			if issue_reportedby == "default":
				default_value = np.zeros(len(le_reportedby_int.classes_),dtype='int').tolist()
				issue_reportedby__ = [default_value]
			else:
				issue_reportedby_ = le_reportedby_int.transform([issue_reportedby]).reshape(-1,1)
				issue_reportedby__ = ce_reportedby.transform(issue_reportedby_).todense().tolist()
			issue_reportedby_final = pca_reportedby.transform(issue_reportedby__)
		
			if issue_affectedperson == "default":
				default_value = np.zeros(len(le_affectedperson_int.classes_),dtype='int').tolist()
				issue_affectedperson__ = [default_value]
			else:
				issue_affectedperson_ = le_affectedperson_int.transform([issue_affectedperson]).reshape(-1,1)
				issue_affectedperson__ = ce_affectedperson.transform(issue_affectedperson_).todense().tolist()
			issue_affectedperson_final = pca_affectedperson.transform(issue_affectedperson__)

			if issue_ownergroup == "default":
				default_value = np.zeros(max_ownergroup_vector_width,dtype='int').tolist()   
				issue_ownergroup__ = [default_value]
			else:
				issue_ownergroup_ = le_ownergroup_int.transform([issue_ownergroup]).reshape(-1,1)
				issue_ownergroup__ = ce_ownergroup.transform(issue_ownergroup_).todense().tolist()
			issue_ownergroup_final = issue_ownergroup__

			if issue_siteid == "default":
				default_value = np.zeros(max_siteid_vector_width,dtype='int').tolist()   
				issue_siteid__ = [default_value]
			else:
				issue_siteid_ = le_siteid_int.transform([issue_siteid]).reshape(-1,1)
				issue_siteid__ = ce_siteid.transform(issue_siteid_).todense().tolist()
			issue_siteid_final = issue_siteid__

			if issue_assetid == "default":
				default_value = np.zeros(max_assetid_vector_width,dtype='int').tolist()   
				issue_assetid__ = [default_value]
			else:
				issue_assetid_ = le_assetid_int.transform([issue_assetid]).reshape(-1,1)
				issue_assetid__ = ce_assetid.transform(issue_assetid_).todense().tolist()
			issue_assetid_final = issue_assetid__
		
			arr_vec = issue_description_final.tolist()[0]
			arr_vec.extend(issue_createdby_final.tolist()[0])
			arr_vec.extend(issue_reportedby_final.tolist()[0])
			arr_vec.extend(issue_affectedperson_final.tolist()[0])
			arr_vec.extend(issue_ownergroup_final[0])
			arr_vec.extend(issue_siteid_final[0])
			arr_vec.extend(issue_assetid_final[0])
			
			arr_vec = np.array(arr_vec).reshape(1,-1)
			class_ = kmeans.predict(arr_vec)[0]
			print("Predicted Class {}".format(class_))

			dataset_IDXs_for_class = kmeans_clusters['cluster_dataset'][class_]
			class_specific_dataset = dataset_X[dataset_IDXs_for_class]
			distance_matrix=euclidean_distances(arr_vec,class_specific_dataset)

			id_=dataset_IDXs_for_class[distance_matrix.argmin()]
			arr=model.docvecs.most_similar(id_)

			similar_sent = data[id_,0]
			t.add_row([str(id_),similar_sent.strip()])
			count=0
			for ar in arr:
				idx = int(ar[0])
				similar_sent = data[idx,0]
				t.add_row([str(idx),similar_sent.strip()])
				count += 1
				if count > 3:
					break
			print(t)
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
			print(e)
