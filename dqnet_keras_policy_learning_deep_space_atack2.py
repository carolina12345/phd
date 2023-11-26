import gym
#from dqnet_cgpt_v3 import DQNAgent
from tensorflow.keras.layers import *
import numpy as np
import tensorflow.keras as layers
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal



# env = gym.make('SpaceInvaders-v0')

# state_size = env.observation_space.shape
# action_size = env.action_space.n
#agent = DQNAgent(state_size, action_size) #v3

#agent = DQNAgent(env) #v4

batch_size = 32
num_episodes = 1000
timesteps_per_episode = 1000


# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

# env.seed(seed)

# state_size = env.observation_space.shape
# num_actions = env.action_space.n

num_actions = 52


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = Input(shape=(210, 160, 3))

    layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer3)

    #reshaped_layer4 = layers.Reshape((6, 6*64))(layer4)
    layer5 = Flatten()(layer4)

    # Reshape the 2D tensor to a 1D sequence
    reshape = Reshape((-1,1))(layer5)


    #sequence_one_hot = Input(shape=(alph_size, alph_size))

    # 1st CNN layer with max-pooling
    conv1 = Convolution1D(256,7,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(reshape)#sequence_one_hot
    pool1 = MaxPooling1D(pool_size=3)(conv1)

    # 2nd CNN layer with max-pooling
    conv2 = Convolution1D(256,7,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=3)(conv2)

    # 3rd CNN layer without max-pooling
    conv3 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(pool2)

    # 4th CNN layer without max-pooling
    conv4 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv3)

    # 5th CNN layer without max-pooling
    conv5 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv4)

    # 6th CNN layer with max-pooling
    conv6 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv5)
    pool6 = MaxPooling1D(pool_size=3)(conv6)

    # Reshaping to 1D array for further layers
    flat = Flatten()(pool6) 

    # 1st fully connected layer with dropout
    dense1 = Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(flat)
    dropout1 = Dropout(0.5)(dense1)

    # 2nd fully connected layer with dropout
    dense2 = Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)

    # 3rd fully connected layer with softmax outputs
    dense3 = Dense(num_actions, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='softmax')(dropout2)

    model = tf.keras.Model(inputs=inputs, outputs=dense3)

    return model


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()


# # In the Deepmind paper they use RMSProp however then Adam optimizer
# # improves training time
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# # Experience replay buffers
# action_history = []
# state_history = []
# state_next_history = []
# rewards_history = []
# done_history = []
# episode_reward_history = []
# running_reward = 0
# episode_count = 0
# frame_count = 0
# # Number of frames to take random action and observe output
# epsilon_random_frames = 50000
# # Number of frames for exploration
# epsilon_greedy_frames = 1000000.0
# # Maximum replay length
# # Note: The Deepmind paper suggests 1000000 however this causes memory issues
# max_memory_length = 100000
# # Train the model after 4 actions
# update_after_actions = 4
# # How often to update the target network
# update_target_network = 10000
# # Using huber loss for stability
# loss_function = tf.keras.losses.Huber()

# while True:  # Run until solved
#     state = np.array(env.reset())
#     episode_reward = 0

#     for timestep in range(1, max_steps_per_episode):
#         # env.render(); Adding this line would show the attempts
#         # of the agent in a pop up window.
#         frame_count += 1

#         # Use epsilon-greedy for exploration
#         if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
#             # Take random action
#             action = np.random.choice(num_actions)
#         else:
#             # Predict action Q-values
#             # From environment state
#             state_tensor = tf.convert_to_tensor(state)
#             state_tensor = tf.expand_dims(state_tensor, 0)
#             action_probs = model(state_tensor, training=False)
#             # Take best action
#             action = tf.argmax(action_probs[0]).numpy()

#         # Decay probability of taking random action
#         epsilon -= epsilon_interval / epsilon_greedy_frames
#         epsilon = max(epsilon, epsilon_min)

#         # Apply the sampled action in our environment
#         state_next, reward, done, _ = env.step(action)
#         state_next = np.array(state_next)

#         episode_reward += reward

#         print("reward", reward)
#         print("episode reward", episode_reward)

#         # Save actions and states in replay buffer
#         action_history.append(action)
#         state_history.append(state)
#         state_next_history.append(state_next)
#         done_history.append(done)
#         rewards_history.append(reward)
#         state = state_next

#         # Update every fourth frame and once batch size is over 32
#         if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

#             # Get indices of samples for replay buffers
#             indices = np.random.choice(range(len(done_history)), size=batch_size)

#             # Using list comprehension to sample from replay buffer
#             state_sample = np.array([state_history[i] for i in indices])
#             state_next_sample = np.array([state_next_history[i] for i in indices])
#             rewards_sample = [rewards_history[i] for i in indices]
#             action_sample = [action_history[i] for i in indices]
#             done_sample = tf.convert_to_tensor(
#                 [float(done_history[i]) for i in indices]
#             )

#             # Build the updated Q-values for the sampled future states
#             # Use the target model for stability
#             future_rewards = model_target.predict(state_next_sample)
#             # Q value = reward + discount factor * expected future reward
#             updated_q_values = rewards_sample + gamma * tf.reduce_max(
#                 future_rewards, axis=1
#             )

#             # If final frame set the last value to -1
#             updated_q_values = updated_q_values * (1 - done_sample) - done_sample

#             # Create a mask so we only calculate loss on the updated Q-values
#             masks = tf.one_hot(action_sample, num_actions)

#             with tf.GradientTape() as tape:
#                 # Train the model on the states and updated Q-values
#                 q_values = model(state_sample)

#                 # Apply the masks to the Q-values to get the Q-value for action taken
#                 q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
#                 # Calculate loss between new Q-value and old Q-value
#                 loss = loss_function(updated_q_values, q_action)

#             # Backpropagation
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))

#         if frame_count % update_target_network == 0:
#             # update the the target network with new weights
#             model_target.set_weights(model.get_weights())
#             # Log details
#             template = "running reward: {:.2f} at episode {}, frame count {}"
#             print(template.format(running_reward, episode_count, frame_count))

#         # Limit the state and reward history
#         if len(rewards_history) > max_memory_length:
#             del rewards_history[:1]
#             del state_history[:1]
#             del state_next_history[:1]
#             del action_history[:1]
#             del done_history[:1]

#         if done:
#             break

#     # Update running reward to check condition for solving
#     episode_reward_history.append(episode_reward)
#     if len(episode_reward_history) > 100:
#         del episode_reward_history[:1]
#     running_reward = np.mean(episode_reward_history)

#     episode_count += 1

#     if running_reward > 40:  # Condition to consider the task solved
#         print("Solved at episode {}!".format(episode_count))
#         break
