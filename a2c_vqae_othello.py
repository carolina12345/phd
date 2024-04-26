
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from functools import reduce
import operator
from keras import backend as K
import gym

frame_skip = 1

env = gym.make("Pong-v0")#, obs_type="grayscale")

num_actions = 3
num_hidden = 128


# Configuration parameters for the whole setup
seed = 42
gamma = 0.5#0.99  # Discount factor for past rewards
max_steps_per_episode = 2*5
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


initial_epsilon = 0.1
decay_factor = 0.99

eps_action = initial_epsilon
eps_position = initial_epsilon

maxlen = num_actions
#############################################################################################################

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        # encoding_indices = self.get_code_indices(flattened)
        # encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        # quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        # quantized = tf.reshape(quantized, input_shape)

        encoding_indices = self.get_code_indices(flattened)
        #encoding_indices = tf.cast(encoding_indices, dtype=tf.float32)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # # Calculate vector quantization loss and add that to the layer. You can learn more
        # # about adding losses to different layers here:
        # # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # # the original paper to get a handle on the formulation of the loss function.
        # commitment_loss = self.beta * tf.reduce_mean(
        #     (tf.stop_gradient(quantized) - x) ** 2
        # )
        # codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        # self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized
    
    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    

    # def get_code_indices(self, flattened_inputs):
    #     # Calculate L2-normalized distance between the inputs and the codes.
    #     similarity = tf.matmul(flattened_inputs, self.embeddings)
    #     distances = (
    #         tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
    #         + tf.reduce_sum(self.embeddings**2, axis=0)
    #         - 2 * similarity
    #     )

    #     # Derive the indices for minimum distances.
    #     encoding_indices = tf.argmin(distances, axis=1)
    #     return encoding_indices




# def get_encoder(inputs, head_size=128, num_heads=4, ff_dim=4, dropout=0.25):
#     # Attention and Normalization
#     x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
#     x = layers.Dropout(dropout)(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     res = x + inputs

#     # Feed Forward Part
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     encoder_outputs = x + res

#     return encoder_outputs
    

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, self.embed_dim)
        self.enc_layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)
        
    def call(self, inputs, training=True):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        embedded = self.embedding(inputs)  # (batch_size, input_seq_len, embed_dim)
        embedded *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        embedded += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(embedded, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'input_vocab_size': self.input_vocab_size,
            'maximum_position_encoding': self.maximum_position_encoding,
            'rate': self.rate,
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                    np.arange(d_model)[np.newaxis, :],
                                    d_model)
      
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
      
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates


def get_decoder(input_shape, output_shape, mlp_units=[128], mlp_dropout=0.4):
    #inputs = keras.Input(shape=build_encoder_model(input_shape).output.shape[1:])
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(output_shape, activation="relu")(x)

    #critic_outputs = layers.Dense(1, activation="relu")(x)
    # x = tf.reshape(x, (-1, maxlen, num_actions))
    # outputs = tf.argmax(x, axis=2)
    return keras.Model(inputs, outputs)


def build_encoder_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0, mlp_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = keras.backend.expand_dims(inputs, axis=-1)
    for _ in range(num_transformer_blocks):
        x = get_encoder(x, head_size, num_heads, ff_dim, dropout)

    #output = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    output = x

    return keras.Model(inputs, output)

#######transformer position############################

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        super(TransformerBlock, self).__init__()

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        if batch_size is None:
            batch_size = 1
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        #causal_mask = tf.expand_dims(causal_mask, axis=1)
        #causal_mask = tf.transpose(causal_mask, (0, 2, 1))
        print('causal_mask', causal_mask.shape)

        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        

        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement an embedding layer
Create two seperate embedding layers: one for tokens and one for token index
(positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        #return positions


def get_vqvae(output_dim, latent_dim=num_hidden, num_embeddings=maxlen, input_shape=maxlen):
    vq_layer = VectorQuantizer(latent_dim, num_embeddings, name="vector_quantizer")

    embed_dim = 64
    num_heads = 4
    ff_dim = 64
    rate = 0.1

    encoder = TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate)

    decoder = get_decoder((49, 36, 64), output_dim*4)


    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)

    #x = layers.Flatten()(x)

    print('x.shape' , x.shape)

    encoder_outputs = encoder(x)
    quantized_latents = vq_layer(encoder_outputs)
    print('quantized_latents.shape', quantized_latents.shape)

    common = decoder(quantized_latents)

    common = layers.Flatten()(common)

    action = layers.Dense(output_dim, activation="linear")(common)
    critic = layers.Dense(1)(common)

    model = keras.Model(inputs, outputs=[action, critic], name="vq_vae")

    return model


state = env.reset()[0]

model_action = get_vqvae(num_actions, input_shape=state.shape)
model_action.compile()

# def sample_from_categorical(logits, temperature=1.0):
#     # Add Gumbel noise
#     gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(shape=tf.shape(logits))))
#     logits_with_noise = (logits + gumbel_noise) / temperature

#     # Apply softmax to obtain differentiable samples
#     samples = tf.nn.softmax(logits_with_noise, axis=-1)

#     return samples

"""
## Train
"""

#optimizer = keras.optimizers.Adam(learning_rate=0.01)
optimizer = keras.optimizers.Adam()#(clipnorm=1.0)

huber_loss = keras.losses.Huber()

action_probs_history = []
position_probs_history = []
critic_value_history = []
critic_pos_value_history = []
action_pos_probs_history = []
critic_value_pos_history = []

action_probs_history_next = []
position_probs_history_next = []
critic_value_history_next = []
critic_pos_value_history_next = []

action_pos_probs_history_next = []
critic_value_pos_history_next = []

epsilon_random_frames = 50000
# Number of frames to take random action and observe output
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken


rewards_history = []
running_reward = 0
episode_count = 0



frame_count = 0

while True:  # Run until solved
    state = env.reset()[0]
    episode_reward = 0

    with tf.GradientTape() as tape:
        for timestep in range(0, max_steps_per_episode):

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            """
            #if frame_count % frame_skip == 0:
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
                action = np.random.choice(num_actions)
            else:                
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.

                # Predict action probabilities and estimated future rewards
                # from environment state
                logits, critic_value = model_action(state)
                #print('logits', logits)
                action_probs = tf.nn.softmax(tf.squeeze(logits), axis=-1).numpy()
                print('action probs', action_probs)

                # Sample action from action probability distribution
                action = np.random.choice(num_actions, 1, p=np.squeeze(action_probs))[0]

                #action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                critic_value_history.append(critic_value[0, 0])
                action_probs_history.append(logits)

            #print(np.squeeze(action_probs), 'action_probs')
            print(action, 'action')
            
            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
            """

            logits, critic_value = model_action(state)
            #print('logits', logits)
            action_probs = tf.nn.softmax(tf.squeeze(logits), axis=-1).numpy()
            
            print('action_probs', action_probs.shape)
            # Sample action from action probability distribution
            action = np.random.choice(num_actions, 1, p=np.squeeze(action_probs))[0]

            print(action, 'action')

            #action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            critic_value_history.append(critic_value[0, 0])
            action_probs_history.append(logits)


            # Apply the sampled action in our environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frame_count += 1
            


            for _ in range(2): #range(frame_skip):
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                frame_count += 1

            if done:
                break

            print('next_obs', next_obs.shape)

            reward = (reward+1)*100
            
            rewards_history.append(reward)
            episode_reward += reward

            #calculate advantage
            action_probs_next, critic_value_next = model_action(tf.expand_dims(next_obs, axis=0), training=False)
            #store values
            action_probs_history_next.append(action_probs_next)
            critic_value_history_next.append(critic_value_next)

            state_next = next_obs
            state = state_next

            print('reward=', reward)


            # else:
            #     next_obs, reward, terminated, truncated, info = env.step(action)
            #     state_next = next_obs
            #     done = terminated or truncated
            #     if done:
            #         break

            

        ## Update running reward to check condition for solving
        #running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()


        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, \
                    returns,\
                    critic_value_history_next)
        actor_losses = []
        critic_losses = []
        for log_prob, critic_value, ret, value_next in history:
            # Calculate advantage
            advantage = ret + gamma * value_next - critic_value
            critic_loss = tf.cast(tf.math.pow(advantage, 2), dtype=tf.double)

            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            log_prob = tf.cast(log_prob, tf.double)
            action = tf.random.categorical(log_prob, num_samples=1) #num_actions

            actor_loss = -tf.cast(action, dtype=tf.double) * tf.cast(advantage, dtype=tf.double)

                ###########################################################################
            #############################################################################

            # # The critic must be updated so that it predicts a better estimate of
            # # the future rewards.
            # critic_losses.append(
            #     huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            # )

            # critic_pos_losses.append(
            #     huber_loss_pos(tf.expand_dims(value_pos, 0), tf.expand_dims(ret, 0))
            # )

            
            # Update Critic
            actor_losses.append(tf.cast(actor_loss, dtype=tf.float32))
            critic_losses.append(tf.cast(critic_loss, dtype=tf.float32))

        #if len(actor_losses)>0 and len(critic_losses)>0:
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model_action.trainable_variables)
        #print(grads)
        optimizer.apply_gradients(zip(grads, model_action.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()

        action_probs_history_next.clear()
        critic_value_history_next.clear()

        rewards_history.clear()

    if done:
        break

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(reward, episode_count))

    # if np.mean(rewards_history) > 21*100:  # Condition to consider the task solved
    #     print("Solved at episode {}!".format(episode_count))
    #     break


print('OUT OF LOOP')


"""
while True:  # Run until solved
    
    episode_reward = 0
    
    for timestep in range(1, max_steps_per_episode):

        with tf.GradientTape() as tape:
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            # Predict action probabilities and estimated future rewards
            # from environment state
            # logits, critic_value = model_action(tf.expand_dims(action, 0))
            # position_logits, critic_pos = model_pos(tf.expand_dims(position, 0))
            logits, position_logits, critic_value = model(state)

            action_probs_history.append(logits)
            position_probs_history.append(position_logits)

            critic_value_history.append(critic_value)



            action_new = tf.argmax(sample_from_categorical(logits), axis=-1)
            position_new = tf.argmax(sample_from_categorical(position_logits), axis=-1)

            # #choose action
            # prob = np.random.rand()
            # if prob < eps_action:
            #     # Exploration: choose a random action
            #     action_new = np.random.choice(num_actions)

                
            # else:
            #     # Exploitation: choose the action with the highest estimated value
            #     #action = tf.random.categorical(logits, num_samples=1)

            #     # Sample action from action probability distribution
            #     #action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            #     action_new = tf.argmax(sample_from_categorical(logits), axis=-1)

            #     #action_probs_history.append(action_probs)

            # #choose position
            # prob = np.random.rand()
            # if prob < eps_position:
            #     # Exploration: choose a random action
            #     position_new = np.random.choice(maxlen)
            # else:
            #     # Exploitation: choose the action with the highest estimated value

            #     # # Sample action from action probability distribution
            #     #position = np.random.choice(maxlen, p=np.squeeze(position_probs))
            #     position_new = tf.argmax(sample_from_categorical(position_logits), axis=-1)
                
            #     #position = tf.random.categorical(position_logits, num_samples=1)

            #     #position_probs_history.append(position_probs)

            # Apply the sampled action in our environment
            print('action', action_new)
            print('position', position_new)
            state_new, reward, done, _ = env.step( (int(action_new), int(position_new)) )

            print('state_new', state_new)

            state_new = tf.reshape(state_new, (1 ,maxlen))

            # # if reward < 20:
            # #     eps_action = min(0.5, eps_action + 0.1)
            # # else:
            # #     eps_action = max(0.1, eps_action - 0.1)            
            # # eps_position = eps_action

            # eps_action *= decay_factor
            # eps_action = max(eps_action, 0.05)
            # eps_position = eps_action

            # #next_state[position] = actiona

            # #print("***********************************next state", next_state) 

            # # with tape_action.stop_recording(), tape_pos.stop_recording():
            # #     next_state = tf.expand_dims(tf.convert_to_tensor(next_state), axis=0)
            # #     action_probs_next, critic_value_next = model_action(next_state)
            # #     position_probs_next, critic_pos_next = model_pos(next_state)

            # # action_probs_next, critic_value_next = tf.stop_gradient(model_action(np.expand_dims(next_state, axis=0)))
            # # position_probs_next, critic_pos_next = tf.stop_gradient(model_pos(np.expand_dims(next_state, axis=0)))
            
            # #with tape_action.stop_recording(), tape_pos.stop_recording():
            # action_logits_next, position_logits_next, critic_value_next = model.predict(tf.convert_to_tensor(state_new))
            #     #action_logits_next, critic_value_next = action_next

            #     #action_probs_next = np.exp(action_logits_next) / np.sum(np.exp(action_logits_next), axis=-1)
            #     #position_logits_next, critic_pos_next = position_next

            #     #position_probs_next = np.exp(position_logits_next) / np.sum(np.exp(position_logits_next), axis=-1)

            # action_probs_history_next.append(action_logits_next)
            # position_probs_history_next.append(position_logits_next)
            # critic_value_history_next.append(critic_value_next)

            # rewards_history.append(reward)
            # episode_reward += reward

            # if done:
            #     break

            # # Update running reward to check condition for solving
            # running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # # Calculate expected value from rewards
            # # - At each timestep what was the total reward received after that timestep
            # # - Rewards in the past are discounted by multiplying them with gamma
            # # - These are the labels for our critic
            # returns = []
            # discounted_sum = 0
            # for r in rewards_history:
            #     discounted_sum = r + gamma * discounted_sum # advantage = reward + (1.0 - done) * gamma * critic(next_state) - critic(state) generate next state
            #     returns.insert(0, discounted_sum)
            
            # # Normalize
            # returns = np.array(returns)
            # returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            # returns = returns.tolist()

            # # Calculating loss values to update our network
            # history = zip(action_probs_history, position_probs_history, critic_value_history, returns, critic_value_history_next)
            # actor_losses = []
            # critic_losses = []
            # position_losses = []
            # critic_pos_losses = []
            # for log_prob, log_prob2, critic_value, ret, critic_value_next in history:
            #     # At this point in history, the critic estimated that we would get a
            #     # total reward = `value` in the future. We took an action with log probability
            #     # of `log_prob` and ended up recieving a total reward = `ret`.
            #     # The actor must be updated so that it predicts an action that leads to
            #     # high rewards (compared to critic's estimate) with high probability.
                
            #     #action_probs, critic_value, position_probs, critic_pos = model.predict(state)

            #     advantage = ret + gamma * critic_value_next - critic_value
            #     critic_loss = tf.cast(tf.math.pow(advantage, 2), dtype=tf.float32)


            #     # Temperature parameter
            #     #log_prob = tf.cast(log_prob, tf.float32)  # Ensure log_prob is float32
            #     # temperature = 1.0
            #     # # Gumbel noise
            #     # gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(shape=tf.shape(log_prob))))
            #     # # Apply Gumbel-Softmax trick
            #     # action_probs = tf.nn.softmax((log_prob + gumbel_noise) / temperature)
            #     # # Sample action from the distribution
            #     # action = tf.argmax(action_probs, axis=-1)
            #     #action = tf.random.categorical(log_prob, num_samples=1) #num_actions

            #     action = tf.argmax(sample_from_categorical(log_prob), axis=-1)

            #     actor_loss = -tf.cast(action, dtype=tf.float32) * tf.cast(advantage, dtype=tf.float32)


            #     #log_prob2 = tf.cast(log_prob2, tf.float32)
            #     # # Temperature parameter
            #     # temperature = 1.0
            #     # # Gumbel noise
            #     # gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(shape=tf.shape(log_prob2))))
            #     # # Apply Gumbel-Softmax trick
            #     # pos_probs = tf.nn.softmax((log_prob2 + gumbel_noise) / temperature)
            #     # # Sample action from the distribution
            #     # position = tf.argmax(pos_probs, axis=-1)
            #     #position = tf.random.categorical(log_prob2, num_samples=1) #maxlen
            #     log_prob2_next = np.exp(log_prob2) / np.sum(np.exp(log_prob2), axis=-1)

            #     position = tf.argmax(sample_from_categorical(log_prob2), axis=-1)

            #     position_loss = -tf.cast(position, dtype=tf.float32) * tf.cast(advantage, dtype=tf.float32)

            #     actor_losses.append(actor_loss)
            #     critic_losses.append(critic_loss)
            #     position_losses.append(position_loss)

            #     ################################################################################

            #     # diff = ret - value
            #     #   # actor loss

            #     # position_losses.append(-log_prob2*diff)

            #     # # The critic must be updated so that it predicts a better estimate of
            #     # # the future rewards.
            #     # critic_losses.append(
            #     #     huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            #     # )

            # state = state_new

            # # Backpropagation
            # #loss_value = sum(actor_losses) + sum(critic_losses) + sum(position_losses) + sum(critic_pos_losses)


        loss = tf.add_n(actor_losses) + tf.add_n(critic_losses) + tf.add_n(position_losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Clear the loss and reward history
    action_probs_history.clear()
    position_probs_history.clear()
    critic_value_history.clear()
    critic_value_history_next.clear()
    rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 100:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

"""










###########################################

# suppose everything have the correct type
# the term 'done' is important because for the end of the episode we only want
# the reward, without the discounted next state value.
