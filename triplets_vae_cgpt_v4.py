from socket import AF_UNIX
import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda, Dense, Flatten, Reshape, Concatenate
from keras.models import Model
from keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
import tensorflow.keras as keras
import tensorflow as tf

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

# Custom triplet loss
def triplet_loss(alpha=0.2):
    def loss(y_true, y_pred, alpha=0.2):
        a, p, n = y_pred[:, :latent_dim], y_pred[:, latent_dim:2*latent_dim], y_pred[:, 2*latent_dim:]

        # Assuming you have some tensor 'x' and a condition 'condition'
        # Calculate Euclidean distances
        dist1 = K.sum(tf.norm(a - p, axis=-1))  # Euclidean distance between anchor and positive
        dist2 = K.sum(tf.norm(a - n, axis=-1))  # Euclidean distance between anchor and negative
        
        # Create a mask indicating which distances need to be swapped        
        # Use tf.where to swap entries based on the mask
        pos_dist  = tf.where(dist1 > dist2, n, p)
        neg_dist  = tf.where(dist1 > dist2, p, n)
        #sort elements in triplet

        # Triplet loss calculation
        loss = K.maximum(0.0, pos_dist - neg_dist + alpha)
        
        return K.mean(loss)
    return loss

# Reparameterization trick
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

# Define TVAE architecture

# Encoder
original_dim = 28 * 28
intermediate_dim = 64
latent_dim = 2

def define_VAE(inputs_out):
    h = keras.layers.Dense(intermediate_dim, activation='relu')(inputs_out)
    z_mean = keras.layers.Dense(latent_dim, trainable=True)(h)
    z_log_sigma = keras.layers.Dense(latent_dim, trainable=True)(h)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    # Create encoder
    encoder = keras.Model(inputs_out, [z_mean, z_log_sigma, z])

    # Create decoder
    latent_inputs = keras.layers.Input(shape=(latent_dim,))
    x = keras.layers.Dense(intermediate_dim, activation='relu', trainable=True)(latent_inputs)
    outputs = keras.layers.Dense(original_dim, activation='sigmoid', trainable=True)(x)
    decoder = keras.Model(latent_inputs, outputs)


    # instantiate VAE model
    encoder_outputs = encoder(inputs_out)

    outputs = decoder(encoder(inputs_out)[2])
    vae = keras.Model(inputs_out, outputs=[encoder_outputs, outputs]) #[[z_mean, z_log_sigma, z], [image]]

    return vae
############# TVAE definition ####################
# from keras.models import clone_model
inputs_a = keras.Input(shape=(original_dim,) )
inputs_p = keras.Input(shape=(original_dim,) )
inputs_n = keras.Input(shape=(original_dim,) )

vae_anchor = define_VAE(inputs_a)

vae_positive = define_VAE(inputs_p)

vae_negative = define_VAE(inputs_n)




TVAE = keras.Model(inputs=[inputs_a, inputs_p, inputs_n], 
                    outputs=[vae_anchor.output,             #[[[z_mean, z_log_sigma, z], [image]], #[[z_mean, z_log_sigma, z], [image]], #[[z_mean, z_log_sigma, z], [image]],  ]
                            vae_positive.output,
                            vae_negative.output
                        ], name='tvae_mlp')



def combined_loss(y_true, y_pred):
    
    # Assuming y_true is a tensor with the same shape as y_pred
    #z_mean, z_log_sigma, z

    #[[[z_mean, z_log_sigma, z], [image]], #[[z_mean, z_log_sigma, z], [image]], #[[z_mean, z_log_sigma, z], [image]],  ]
    latent_mean_anchor, z_log_sigma_anchor, z = y_pred[0][0]
    y_pred_anchor = y_pred[0][1]

    latent_mean_positive, z_log_sigma_positive, z = y_pred[1][0]
    y_pred_positive = y_pred[1][1]

    latent_mean_negative, z_log_sigma_negative, z = y_pred[2][0]
    y_pred_negative = y_pred[2][1]

    y_true_a = y_true[0]#[0,:,:]
    y_true_p = y_true[1]#[1,:,:]
    y_true_n = y_true[2]#[2,:,:]

    # triplet loss
    # Concatenate the triplet outputs
    triplet_outputs = K.concatenate([latent_mean_anchor, latent_mean_positive, latent_mean_negative], axis=1)

    # binary cross entropy loss
    bce_loss = K.mean(K.binary_crossentropy(y_true_a, y_pred_anchor), axis=-1) + K.mean(K.binary_crossentropy(y_true_p, y_pred_positive), axis=-1) + K.mean(K.binary_crossentropy(y_true_n, y_pred_negative), axis=-1)

    # KL Divergence loss
    kl_loss_anchor = -0.5 * K.sum(1 + z_log_sigma_anchor - K.square(latent_mean_anchor) - K.exp(z_log_sigma_anchor), axis=-1)
    kl_loss_positive = -0.5 * K.sum(1 + z_log_sigma_positive - K.square(latent_mean_positive) - K.exp(z_log_sigma_positive), axis=-1)
    kl_loss_negative = -0.5 * K.sum(1 + z_log_sigma_negative - K.square(latent_mean_negative) - K.exp(z_log_sigma_negative), axis=-1)
    kl_loss = kl_loss_anchor + kl_loss_positive + kl_loss_negative

    return K.sum(bce_loss + K.mean(kl_loss) + triplet_loss()(None, triplet_outputs))


# Train
batch_size = 32
x_train = np.reshape(x_train, (-1, original_dim, 3))

#x_train = [x_train[:,:,0], x_train[:,:,1], x_train[:,:,2] ]

# Choose an optimizer (e.g., Adam) and specify learning rate if needed
optimizer = Adam(learning_rate=0.001)

x_train_batched = np.reshape(x_train, (-1, batch_size, original_dim, 3))

n_epochs = 50



for epoch in range(n_epochs):

    total_loss = 0

    for n_batch in range(x_train_batched.shape[0]):
        with tf.GradientTape() as tape:
            batch_a = x_train_batched[n_batch,:,:,0]
            batch_p = x_train_batched[n_batch,:,:,1]
            batch_n = x_train_batched[n_batch,:,:,2]

            batch_x = [batch_a, batch_p, batch_n]

            predictions = TVAE(batch_x)
            loss = combined_loss(batch_x, predictions)
            total_loss = total_loss + loss
        
        gradients = tape.gradient(loss, TVAE.trainable_variables)
        optimizer.apply_gradients(zip(gradients, TVAE.trainable_variables))

    print('epoch ', epoch,' , loss=', total_loss)



# # Train the model
# TVAE.fit(x_train, epochs=5, batch_size=64)

TVAE.save("tvae_harvard.h5")
# vae.save("vae_harvard.h5")
# encoder.save("encoder_harvard.h5")
# decoder.save("decoder_harvard.h5")


################################################################################

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# Assuming `x_test` is your testing dataset
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


x_test_reshaped = np.reshape(x_test[0:1110], (-1, original_dim, 3))
y_test_reshaped = np.reshape(y_test[0:1110], (-1, 3))

# bce_input_a = x_train_reshaped[:,:,0]
# bce_input_p = x_train_reshaped[:,:,1]
# bce_input_n = x_train_reshaped[:,:,2]

# Get reconstruction predictions
reconstructed_data = TVAE.predict( [x_test_reshaped[:,:,0], x_test_reshaped[:,:,1], x_test_reshaped[:,:,2] ])

#[vae_anchor.ouputs, vae_positive.ouputs, vae_negative.ouputs]
#[encoder_outputs, outputs]
# Assuming `encoder1`, `encoder2`, and `encoder3` are the encoders of your three TVAE models
encoded_data1, _, _ = reconstructed_data[0][0]
encoded_data2, _, _  = reconstructed_data[1][0]
encoded_data3, _, _  = reconstructed_data[2][0]


# latent_mean_anchor, z_log_sigma_anchor, z = y_pred[0][0]
# y_pred_anchor = y_pred[0][1]
# latent_mean_positive, z_log_sigma_positive, z = y_pred[1][0]
# y_pred_positive = y_pred[1][1]
# latent_mean_negative, z_log_sigma_negative, z = y_pred[2][0]
# y_pred_negative = y_pred[2][1]

# Assuming `y_test` contains the true labels of your testing data
# You can use this for color-coding the t-SNE plots
tsne = TSNE(n_components=2, random_state=42)
tsne_results1 = tsne.fit_transform(encoded_data1)
tsne_results2 = tsne.fit_transform(encoded_data2)
tsne_results3 = tsne.fit_transform(encoded_data3)

# Visualize the t-SNE plots for each TVAE
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(tsne_results1[:, 0], tsne_results1[:, 1], c=y_test_reshaped[:,0], cmap='viridis')
plt.title('t-SNE Visualization - TVAE 1')
plt.savefig('tvae1')

plt.subplot(1, 3, 2)
plt.scatter(tsne_results2[:, 0], tsne_results2[:, 1], c=y_test_reshaped[:,1], cmap='viridis')
plt.title('t-SNE Visualization - TVAE 2')
plt.savefig('tvae2')

plt.subplot(1, 3, 3)
plt.scatter(tsne_results3[:, 0], tsne_results3[:, 1], c=y_test_reshaped[:,2], cmap='viridis')
plt.title('t-SNE Visualization - TVAE 3')
plt.savefig('tvae3')