import gym
from gym import spaces
import numpy as np

from net_builder_googleNet_sequence import *
from transformer_keras_io import *
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import model_from_json

class NASEnvironment(gym.Env):
    def __init__(self, 
                train_features, train_labels, 
                validation_features, validation_labels, 
                eval_features, eval_labels, 
                epochs, sequence_len):
        super(NASEnvironment, self).__init__()

        self.sequence_len = sequence_len

        self.vocab_size = 20000
        self.embedding_dim = 32
        self.max_length = 250

        self.batch_size = 32
        #initial network parameters
        self.input_shape = (self.max_length)  # (batch_size, input_length, input_channels)
        self.num_classes = 2
        
        self.cut_value = 2

        # Define action and observation space
        self.action_space = spaces.Discrete(self.sequence_len)  # Two possible actions
        self.observation_space = spaces.Discrete(self.sequence_len)  # Observation space with 3 features

        # Initialize state
        self.state = np.zeros(self.sequence_len)  # Initial state with three features

        #training parameters
        self.train_features = np.array(train_features)
        self.train_labels = np.array(train_labels, dtype=np.int32)
        self.validation_features = np.array(validation_features)
        self.validation_labels = np.array(validation_labels, dtype=np.int32)
        self.eval_features = np.array(eval_features)
        self.eval_labels = np.array(eval_labels, dtype=np.int32)
        self.epochs = epochs

        #stopping condition
        self.satisfaction_limit = 90

        self.last_reward = 0
        self.last_state = np.zeros(self.sequence_len)

    def reset(self):
        # Reset the environment to its initial state
        self.state = np.zeros(self.sequence_len)
        return self.state

    def step(self, action):
        val_action, pos_action = action

        self.state[pos_action] = val_action

        if all(self.state == self.last_state):
            return self.state, self.last_reward, False, {}

        # Initialize root node with Conv1D parameters
        #cut state sequence where it meets value
        try:
            cut_index = np.where(self.state == self.cut_value)[0][0]
            cut_result = self.state[:cut_index-1]
        except:
            cut_result = self.state

        try:

            model = build_tree_model(self.input_shape, self.num_classes, cut_result, self.max_length, self.vocab_size, self.embedding_dim)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            
            try:
                history = model.fit(
                    self.train_features, self.train_labels, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.validation_features, self.validation_labels)
                )
                
                # Access training accuracy from the history object
                training_accuracy = history.history['accuracy']
            except Exception as e:
                print(f"An error occurred: {e}")
                training_accuracy = 0


            #model evaluation
            # predicted_labels = model.predict(self.eval_features)
            # reward = accuracy_score(self.eval_labels, predicted_labels)*100
                
            reward = np.mean(training_accuracy)*100
            print('Current reward: ', reward)

            # Return the new state, reward, whether the episode is done, and additional info
            if reward>self.satisfaction_limit:
                done = True
                # Save the model architecture to JSON
                model_json = model.to_json()
                with open('final_model.json', 'w') as json_file:
                    json_file.write(model_json)
                model.save('final_model.h5')
                print(self.state)
            else:
                done = False

            self.last_reward = reward
            self.last_state = self.state

            return self.state, reward, done, {} #info
        except:
            self.last_state = self.state
            self.last_reward = 0

            return self.state, 0, False, {}

# TODO: saving architectures logic while training
