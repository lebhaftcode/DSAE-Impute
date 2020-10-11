from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

data = pd.read_csv('data/jurkat/raw.csv', index_col=0)
adj = cosine_similarity(data.values)
print(adj)

