from tensortango import wine_data_reader, train, layer, mlp 

df = wine_data_reader.read()
labels = df['quality']
features = df[[col for col in df.columns if col != 'quality']]

# Min max scaling -- maybe you want to 
for col in features:
    features[col] -= features[col].min()
    features[col] /= features[col].max() # equivalent to features[col] = features[col]/

neural_net = mlp.MLP()