import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.initializers import HeNormal
import matplotlib.pyplot as plt
import folium
import geopandas as gpd


# Load NRI data
df = pd.read_csv('Proj2_VAE/NRI_Table_Counties/NRI_Table_Counties.csv')
nri_data = pd.read_csv('Proj2_VAE/nri_data.csv')

x_train, x_test = train_test_split(nri_data, test_size=0.2, random_state=42)

# Convert the data type of the NumPy array to a supported type; float32:
x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)

if np.isnan(x_train).any() or np.isnan(x_test).any():
    print("Input data contains NaN values.")


# Network parameters
original_dim = x_train.shape[1]
latent_dim = 2
intermediate_dim = 256

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(original_dim,), name='encoder_input')
x = Dense(intermediate_dim, kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01), activation='relu')(inputs)
x = BatchNormalization()(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Next, we need to define a function for sampling from the latent space:
# reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01), activation='relu')(latent_inputs)
x = BatchNormalization()(x)
outputs = Dense(original_dim, kernel_initializer=HeNormal(), activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# compile and train the VAE model
# define a custom loss function that includes both the reconstruction loss and the KL divergence:

reconstruction_loss = original_dim * metrics.binary_crossentropy(inputs, outputs)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Use a smaller learning rate
optimizer = Adam(learning_rate=0.001)
vae.compile(optimizer=optimizer, loss = None)

n_batch_sizes = 5

history = ['']*n_batch_sizes
titles = ['']*n_batch_sizes

#train the VAE
# alternate between few epochs w many batch sizes (less loss but more likely to overfit)
# and many epoch with small batch (strong and non-overfitting but could increase loss)
for i in range(n_batch_sizes):
    batch_size = 512*(1-(i%2)) + 16*(i%2)
    epochs = 20*(1-(i%2)) + 100*(i%2)
    
    print('Batch size:', batch_size)
    titles[i] = 'Round {}, Batch size {}'.format(i+1, batch_size)
    history[i] = vae.fit(x = x_train, y = None, shuffle = True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

'''
# graph log loss
f, ax = plt.subplots(len(history), 1, figsize = (7, len(history)*4), sharex = True)
for i in range (len(history)):
    ax[i].set_title(titles[i])
    ax[i].plot(history[i].epoch, np.log(history[i].history['val_loss']),label='Validation')
    ax[i].plot(history[i].epoch, np.log(history[i].history['loss']),label='Training')
    ax[i].autoscale()
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('Log loss')
    ax[i].legend()
    ax[i].grid()
    directory_path = 'Proj2_VAE/Log_Loss_Graphs/'
    title = titles[i]
    fig_path = directory_path + f"{title}.png"
    plt.savefig(fig_path)
'''


# Use the trained VAE model to generate predictions
z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
predictions = decoder.predict(z_mean)
print('done')

if np.isnan(predictions).any():
    print("Model's predictions contain NaN values.")

print("predictions shape: ", predictions.shape) 


# create a new visualization
nri_data1 = np.asarray(nri_data).astype(np.float32)
z_mean1, _, _ = encoder.predict(nri_data1, batch_size=batch_size)
pred1 = decoder.predict(z_mean1)
print(pred1)

# exctract column14 (RISK_SCORE)
column14 = pred1[:, 14]
print(column14)

# Create a DataFrame with predicted values
column14 = pd.DataFrame(column14)
column14.columns = ['RISK']
column14['COUNTY'] = df['COUNTY'].str.upper()
# Identify quintile for feature "column14" in df and create a new variable "RISK" that indicates which quintile each entry falls under.
column14['RISK'] = pd.qcut(column14['RISK'], q=5, labels=False)

column14.to_csv('Proj2_VAE/column14.csv')


# Create map visualization using Geopandas

# Path to the shapefile of US counties
shapefile_path = 'Proj2_VAE/cb_2018_us_county_5m/cb_2018_us_county_5m.shp'
# Read the shapefile into a GeoDataFrame
gdf_counties = gpd.read_file(shapefile_path)
gdf_counties['COUNTY'] = gdf_counties['NAME'].str.upper()

print(gdf_counties.columns)
print(column14.columns)

gdf_counties = gdf_counties.merge(column14, left_on='COUNTY', right_on='COUNTY', how='left')
print(gdf_counties.columns)
print(gdf_counties.head())

# Check for missing values
print("missing values check: ", gdf_counties['RISK'].isnull().sum())
# Fill missing values with a specific value (e.g., 0)
gdf_counties['RISK'].fillna(0, inplace=True)
# Check the data type of the 'RISK' data
print(gdf_counties['RISK'].dtypes)
# Convert the 'RISK' data to a numerical type if necessary
gdf_counties['RISK'] = pd.to_numeric(gdf_counties['RISK'], errors='coerce')


# Plot the data
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the counties with color-coded RISK
gdf_counties.plot(column='RISK', cmap='viridis', linewidth=0.1, ax=ax, edgecolor='0.8')
# Customize the plot
ax.set_title('NRI Natural Disaster Risk Rating by County')
ax.axis('off')
# Add a colorbar with a smaller size
sm = plt.cm.ScalarMappable(cmap='viridis')
sm.set_array(gdf_counties['RISK'])
cbar = plt.colorbar(sm, ax=ax, fraction=0.02)  # Specify the ax argument
# Show and save the plot
plt.savefig('RISKMAP.png')
plt.show()

'''
# Create a new visualization 
nri_data1 = np.asarray(nri_data).astype(np.float32)
z_mean1, _, _ = encoder.predict(nri_data1, batch_size=batch_size)
pred1 = decoder.predict(z_mean1)
print(pred1)
column0 = pred1[:, 0]
print(column0)

dfpred1 = pd.DataFrame(pred1)
dfpred1.to_csv('Proj2_VAE/pred1.csv')
'''
print("pred1 shape: ", pred1.shape) 
print("column14 shape: ", column14.shape)
print("nri_data shape: ", nri_data.shape)
print("df shape: ", df.shape)

'''
# Add the predictions to the original DataFrame
nri_data['Predicted Risk Level'] = predictions_df['Predicted Risk Level']

# predictions_df['latitude'] = geo_data['latitude']
# predictions_df['longitude'] = geo_data['longitude']


# Create base map
map = folium.Map(location=[40.693943, -73.985880], default_zoom_start=15)

for index, row in predictions.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup='Predicted Risk: {}'.format(row['predicted_risk']),
        tooltip="Click for more"
    ).add_to(map)

map
map.save('risk_prediction_map.html')
'''