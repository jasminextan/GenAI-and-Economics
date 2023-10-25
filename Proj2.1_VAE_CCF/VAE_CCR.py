import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Input, Model
from keras.layers import Dense, Lambda
from keras import backend as K
from keras.losses import mse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import seaborn as sns


# Load the dataset
data = pd.read_csv('Proj2.1_VAE_CCF/creditcard.csv')

# Scale 'Time' and 'Amount' features
scaler = StandardScaler()
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# DEFINE VAE MODEL
input_dim = train_data.shape[1] - 1  # excluding 'Class'
intermediate_dim = 15
latent_dim = 2

inputs = Input(shape=(input_dim,), name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
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

# Use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(input_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# DEFINE LOSS AND COMPILE MODEL
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= input_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# TRAIN MODEL
X_train = train_data.drop('Class', axis=1)
vae.fit(X_train, epochs=50, batch_size=32)

# DETECT ANOMOLIES
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']
X_test_pred = vae.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': y_test})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision")
plt.plot(threshold_rt, recall_rt[1:], label="Recall")
plt.title('Precision-Recall Curve', fontsize=15)
plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
plt.show()
plt.savefig('PRCurve.png')


# Find the optimal threshold value
optimal_idx = np.argmax(np.sqrt(precision_rt[1:] * recall_rt[1:]))
optimal_threshold = threshold_rt[optimal_idx]
print("Threshold value is: ", optimal_threshold)

# EVALUATE 
# Predict the 'Class' scores using the optimal threshold value
pred_y = [1 if e > optimal_threshold else 0 for e in error_df.Reconstruction_error.values]

# Evaluate the model
conf_matrix = confusion_matrix(error_df.True_class, pred_y)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=15)
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.show()
plt.savefig('ConfMatrix.png')
