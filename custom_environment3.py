import gym
from gym import spaces
import numpy as np

from net_builder_minimal_sequence import *
from transformer_keras_io import *
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import model_from_json
import pickle

class NASEnvironment():
    def __init__(self, 
                train_features, train_labels, 
                validation_features, validation_labels, 
                eval_features, eval_labels, 
                epochs, sequence_len):

        self.sequence_len = sequence_len

        self.vocab_size = 20000
        self.embedding_dim = 32
        self.max_length = 250

        self.batch_size = 16
        #initial network parameters
        self.input_shape = train_features[0].shape # (batch_size, input_length, input_channels)
        self.num_classes = 52
        
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

        self.seen_states = []

    def reset(self):
        # Reset the environment to its initial state
        self.state = np.zeros(self.sequence_len)
        return self.state
    

    def find_by_state(self, cur_state):
        for s in self.seen_states:
            if s[0] == cur_state:
                return s
        return None


    def step(self, action):
        val_action, pos_action = action

        self.state[pos_action] = val_action

        # Initialize root node with Conv1D parameters
        #cut state sequence where it meets value
        if np.where(self.state == self.cut_value)[0].size != 0:
            cut_index = np.where(self.state == self.cut_value)[0][0]

            # Split the array into two parts
            first_part = self.state[:cut_index]
            second_part = self.state[cut_index:]
            
            # Concatenate the first part with the padded second part
            self.state = np.concatenate((first_part, 2*np.ones_like(second_part)))

            self.evaluate_state = first_part
        else:
            self.evaluate_state = self.state

        if all(self.state == self.last_state):
            return self.state, self.last_reward, False, {}
        

        # seen_state = self.find_by_state(self.state)
        # if seen_state: #seen state
        #     self.last_reward = seen_state[1]
        #     return self.state, seen_state[1], False, {}  

        try:
            print('building model')
            float_strings = [str(int(float_value)) for float_value in self.evaluate_state]
            print(''.join(float_strings))

            model = build_tree_model(self.input_shape, ''.join(float_strings), self.num_classes)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            
            model.summary()

            history = model.fit(
                self.train_features, self.train_labels, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.validation_features, self.validation_labels)
            )

            #predictions = model.predict(self.eval_features)

            #prdictions_binary = (predictions >= 0.5).astype(int)
            
            # Access training accuracy from the history object
            #training_accuracy = history.history['val_accuracy']
            training_accuracy = history.history['val_accuracy'][-1]*100


            #model evaluation
            # predicted_labels = model.predict(self.eval_features)
            # reward = accuracy_score(self.eval_labels, predicted_labels)*100
                
            #reward = np.mean(training_accuracy)*100
            reward = training_accuracy
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

                # Pickle the list
                with open('winner_state.pkl', 'wb') as f:
                    pickle.dump(self.state, f)
            else:
                done = False

            self.last_reward = reward
            self.last_state = self.state

            self.seen_states.append(tuple(self.state, reward))

            return self.state, reward, done, {} #info
        except Exception as e:
            print('Error when building model: ', e)

            self.last_state = self.state
            self.last_reward = 0

            

            return self.state, 0, False, {}

# TODO: saving architectures logic while training
