import kagglehub

# Run this file to download datasets from different kaggle sources

# path = kagglehub.dataset_download("catiateixeira/wordwide-pm-polution-and-related-mortality")
# path = kagglehub.dataset_download("himanshunakrani/iris-dataset")
# path = kagglehub.dataset_download("rodolfomendes/abalone-dataset")

# more info about this one: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
# path = kagglehub.dataset_download("sohommajumder21/appliances-energy-prediction-data-set")

path = kagglehub.dataset_download("quantbruce/real-estate-price-prediction")

print("Path to dataset files:", path)
