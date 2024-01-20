
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the IMDb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Pad sequences to a fixed length
max_sequence_length = 250
x_train = pad_sequences(x_train, maxlen=max_sequence_length)
x_val = pad_sequences(x_val, maxlen=max_sequence_length)
x_test = pad_sequences(x_test, maxlen=max_sequence_length)

# Print information about the datasets
print(f"Number of training examples: {len(x_train)}")
print(f"Number of testing examples: {len(x_test)}")

# Define the model
"""
Title: Actor Critic Method
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/13
Last modified: 2020/05/13
Description: Implement Actor Critic Method in CartPole environment.
"""
"""
## Introduction
This script shows an implementation of Actor Critic method on CartPole-V0 environment.
### Actor Critic Method
As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to two possible outputs:
1. Recommended action: A probability value for each action in the action space.
   The part of the agent responsible for this output is called the **actor**.
2. Estimated rewards in the future: Sum of all rewards it expects to receive in the
   future. The part of the agent responsible for this output is the **critic**.
Agent and Critic learn to perform their tasks, such that the recommended actions
from the actor maximize the rewards.
### CartPole-V0
A pole is attached to a cart placed on a frictionless track. The agent has to apply
force to move the cart. It is rewarded for every time step the pole
remains upright. The agent, therefore, must learn to keep the pole from falling over.
### References
- [CartPole](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf)
- [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)
"""
"""
## Setup
"""
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from custom_environment2 import NASEnvironment
from custom_environment3 import NASEnvironment

sequence_len = 15
maxlen = sequence_len
num_inputs = maxlen
num_actions = 3
num_hidden = 128


# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 2 #10000
env = NASEnvironment(x_train, y_train, x_val, y_val, x_test, y_test, epochs=7, sequence_len=sequence_len)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


initial_epsilon = 0.1
decay_factor = 0.99

eps_action = initial_epsilon
eps_position = initial_epsilon


num_inputs = 4
num_actions = 3
num_hidden = 128
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


def get_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.25):
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    encoder_outputs = x + res

    return encoder_outputs #keras.Model(inputs, encoder_outputs, name="encoder")


def get_decoder(input_shape, mlp_units=[128], mlp_dropout=0.4):
    #inputs = keras.Input(shape=build_encoder_model(input_shape).output.shape[1:])
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    #outputs = layers.Dense(output_shape, activation="relu")(x)

    critic_outputs = layers.Dense(1, activation="relu")(x)
    # x = tf.reshape(x, (-1, maxlen, num_actions))
    # outputs = tf.argmax(x, axis=2)
    return keras.Model(inputs, critic_outputs)


def build_encoder_model(input_shape, output_dim, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0, mlp_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = keras.backend.expand_dims(inputs, axis=-1)
    for _ in range(num_transformer_blocks):
        x = get_encoder(x, head_size, num_heads, ff_dim, dropout)

    #output = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    output = x

    encoder_outputs = layers.Flatten()(output)
    action = layers.Dense(output_dim)(encoder_outputs)

    critic = layers.Dense(1)(encoder_outputs)

    return keras.Model(inputs, [action, critic])

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


def get_vqvae(input_dim, output_dim):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    vocab_size = 20000  # Only consider the top 20k words

    inputs = layers.Input(shape=(input_dim,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    action = layers.Dense(output_dim, activation="softmax")(x)
    critic = layers.Dense(1, activation="relu")(x)

    model = keras.Model(inputs, outputs=[action, critic], name="vq_vae")

    return model




################################################################################################################

# def get_model_action():
#     inputs = layers.Input(shape=(maxlen,))
#     common = layers.Dense(num_hidden, activation="relu")(inputs)
#     action = layers.Dense(num_actions, activation="softmax")(common)
#     critic = layers.Dense(1)(common)

#     model_action = keras.Model(inputs=inputs, outputs=[action, critic])

#     return model_action

# def get_model_position():
#     inputs = layers.Input(shape=(maxlen,))
#     common = layers.Dense(num_hidden, activation="relu")(inputs)
#     action = layers.Dense(maxlen, activation="softmax")(common)
#     critic = layers.Dense(1)(common)

#     model_position = keras.Model(inputs=inputs, outputs=[action, critic])

#     return model_position

# model_action = get_model_action()
# model_position = get_model_position()

model_action = get_vqvae(maxlen, num_actions)
model_position = get_vqvae(maxlen, maxlen)

def sample_from_categorical(logits, temperature=1.0):
    # Add Gumbel noise
    gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(shape=tf.shape(logits))))
    logits_with_noise = (logits + gumbel_noise) / temperature

    # Apply softmax to obtain differentiable samples
    samples = tf.nn.softmax(logits_with_noise, axis=-1)

    return samples

"""
## Train
"""

#optimizer = keras.optimizers.Adam(learning_rate=0.01)
optimizer = keras.optimizers.Adam()
optimizer_pos = keras.optimizers.Adam()

huber_loss = keras.losses.Huber()
huber_loss_pos = keras.losses.Huber()

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


rewards_history = []
running_reward = 0
episode_count = 0

state = env.reset()

state = tf.convert_to_tensor(state)
state = tf.expand_dims(state, 0)

action = 2
position = 10
#state = np.array([0, 1, 8, 2, 6, 5, 8, 3, 6, 4, 9, 1, 1, 1, 1])


while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape, tf.GradientTape() as tape_pos:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model_action(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))


            # Predict position action probabilities and estimated future rewards
            # from environment state
            action_pos_probs, critic_value = model_position(state)
            critic_value_pos_history.append(critic_value[0, 0])

            action_pos = np.random.choice(maxlen, p=np.squeeze(action_pos_probs))
            action_pos_probs_history.append(tf.math.log(action_pos_probs[0, action_pos]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step((action, action_pos))
            rewards_history.append(reward)
            episode_reward += reward

            #calculate advantage
            action_probs_next, critic_value_next = model_action.predict(tf.convert_to_tensor(tf.expand_dims(state, 0)))
            action_pos_probs_next, critic_value_pos_next = model_position.predict(tf.convert_to_tensor(tf.expand_dims(state, 0)))
            #store values
            action_probs_history_next.append(action_probs_next)
            critic_value_history_next.append(critic_value_next)

            action_pos_probs_history_next.append(action_pos_probs_next)
            critic_value_pos_history_next.append(critic_value_pos_next)

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

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
        history = zip(action_probs_history, action_pos_probs_history, critic_value_history, critic_value_pos_history, \
                      returns,\
                      action_probs_history_next, critic_value_history_next, action_pos_probs_history_next, critic_value_pos_history_next)
        actor_losses = []
        actor_pos_losses = []
        critic_losses = []
        critic_pos_losses = []
        for log_prob, log_prob_pos, value, value_pos, ret, log_prob_next, value_next, log_prob_pos_next, value_pos_next in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            diff_pos = ret - value_pos
            actor_pos_losses.append(-log_prob_pos * diff_pos)

            # # The critic must be updated so that it predicts a better estimate of
            # # the future rewards.
            # critic_losses.append(
            #     huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            # )

            # critic_pos_losses.append(
            #     huber_loss_pos(tf.expand_dims(value_pos, 0), tf.expand_dims(ret, 0))
            # )

            # Calculate advantage
            advantage = reward + gamma * value_next - value
            # Update Critic
            critic_losses.append(tf.math.pow(advantage,2))

            # Calculate advantage
            advantage_pos = reward + gamma * value_pos_next - value_pos
            # Update Critic
            critic_pos_losses.append(tf.math.pow(advantage_pos,2))

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model_action.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_action.trainable_variables))

        loss_value_pos = sum(actor_pos_losses) + sum(critic_pos_losses)
        grads_pos = tape_pos.gradient(loss_value_pos, model_position.trainable_variables)
        optimizer_pos.apply_gradients(zip(grads_pos, model_position.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()

        action_pos_probs_history.clear()
        critic_value_pos_history.clear()

        action_probs_history_next.clear()
        critic_value_history_next.clear()

        action_pos_probs_history_next.clear()
        critic_value_pos_history_next.clear()

        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break


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
