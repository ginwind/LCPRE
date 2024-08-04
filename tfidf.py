import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# modify the datapath and the threhold of TFIDF here.
path = "./dataset/UCD"
threshold = 0.05

cc_edges = pd.read_csv(path + "/cc.csv").drop_duplicates()
rr_edges = pd.read_csv(path + "/rr.csv").drop_duplicates()
resourceDes = pd.read_csv(path + "/courses.csv")

corpus = resourceDes["Descriptions"].drop_duplicates().to_list()
concepts = pd.concat([cc_edges["Concept1"], cc_edges["Concept2"]]).drop_duplicates().to_list()
concept_max_ngram = max([len(each.split()) for each in concepts])
concept_min_ngram = min([len(each.split()) for each in concepts])

print(f"min_ngram:{concept_min_ngram}, max_ngram:{concept_max_ngram}")
print(f"num_resources:{len(corpus)}, num_concepts:{len(concepts)}")
tfidf_vectorizer = TfidfVectorizer(vocabulary=concepts, ngram_range=(concept_min_ngram, concept_max_ngram))

tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_matrix = tfidf_matrix.toarray()
print(f"TF-IDF matrix for the word set:{tfidf_matrix.shape}")

rc_R, rc_C = [], []
for i in tqdm(range(0, tfidf_matrix.shape[0])):
    for j in range(0, tfidf_matrix.shape[1]):
        if tfidf_matrix[i][j] > threshold:
            rc_R.append(resourceDes.iloc[i]["Courses"])
            rc_C.append(concepts[j])
print(len(rc_R))

rc_df = pd.DataFrame({"Concepts": rc_C, "Courses": rc_R})
rc_df.to_csv(path+"/rc.csv", index=False)
    


