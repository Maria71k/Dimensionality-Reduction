from sklearn.decomposition import PCA
import pandas as pd

def dimensionality_reduction(data):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# Example usage:
data = pd.read_csv("high_dimensional_data.csv")
reduced_data = dimensionality_reduction(data)
print(reduced_data)
