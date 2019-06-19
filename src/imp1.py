import tensorflow as tf
import numpy as np
import tflearn
import os, sys
import xml.etree.ElementTree as et

import tools.traci as traci
import tools.tls as tls
import tools.sumolib as sumolib


class Network:

    def __init__(self, no_of_lanes, no_of_layers, juncID, trafficID, connectedE2):
        self.inputs = tf.placeholder(tf.float32)
        self.outputs = tf.placeholder(tf.float32)
        self.juncID = juncID
        self.weights = []
        self.biases = []
        self.nol = no_of_layers
        self.nolane = no_of_lanes
        self.state_buffer = []  # stores the states
        self.q_value_buffer = []  # stores the Q(s,a)
        self.action_index_buffer = []
        self.reward_buffer = []
        self.detectors = connectedE2
        self.trafficId = trafficID
        self.model_object = None

    def get_reward(self):
        reward = 0
        e2 = traci._lanearea.LaneAreaDomain()
        for detector in self.detectors:
            reward += e2.getLastStepHaltingNumber(detector)
        self.reward_buffer.append(reward)

    def updateQTable(self, disco_factor, alpha):  # we need to write reward fn.#added loss function

        def QmaxFinder(i):
            temp = self.q_value_buffer[i + 1]
            return np.max(temp.reshape(2 * self.nolane - 1, 3), axis=1)

        last_index = len(self.action_index_buffer) - 1
        self.get_reward()

        q = self.q_value_buffer[last_index][self.action_index_buffer[last_index - 1]]

        self.q_value_buffer[last_index][self.action_index_buffer[last_index - 1]] += \
            alpha * (self.reward_buffer[last_index] - q)

        while last_index >= 0:
            last_index -= 1
            self.get_reward()
            q = self.q_value_buffer[last_index][self.action_index_buffer[last_index - 1]]

            Qmax = QmaxFinder(last_index)
            self.q_value_buffer[last_index][self.action_index_buffer[last_index - 1]] += alpha * (
                    self.reward_buffer[last_index] + disco_factor * Qmax - q)

    def create_model(self, n):
        net = self.inputs
        net = tflearn.fully_connected(net, 64, activation='tanh', name='net1')  # added names
        net = tflearn.fully_connected(net, 32, activation='tanh', name='net2')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, 32, activation='tanh', name='net3')
        net = tflearn.fully_connected(net, 6 * self.nol - 3, activation='linear', name='net4')
        regression = tflearn.regression(net, optimizer='adam', learning_rate=n, loss='mean_square')
        self.model_object = (tflearn.DNN(regression))

    def store_q_values(self, input_state):
        if self.model_object != None:
            predict_q = self.model_object.predict(input_state)
            self.state_buffer.append(input_state)
            self.q_value_buffer.append(predict_q)

        # changed the defn.#it is actually get Q values..??not giving actions

    def get_action_index(self, eps=1.):
        array = np.array(self.q_value_buffer[-1])
        rows = 2 * self.nolane - 1
        action_index_list = []
        array.reshape([rows, 3])
        for i in range(rows):
            if np.random.uniform(0, 1) > eps:
                action_index_list.append(np.argmax(array[i]))
            else:
                action_index_list.append(np.random.choice([0, 1, 2]))
        self.action_index_buffer.append(np.array(action_index_list))

        required_q_index = np.argmax(array, axis=1) - 1

        a = -1 * np.sum(required_q_index) + 1  # getting action of the (2k)th phase
        return (list(required_q_index + 1) + [a]), np.max(array, axis=1)  # changed the return statement

    def next_state(self, curr_state, delta):
        action, _ = self.get_action_index(0.5)  # take care of gradual decrease of epsilon
        action = np.array(action)
        action = action * delta
        state_next = np.array(curr_state) + action
        return state_next

    '''    def train(self):
            file_name = "variables.tflearn"
            if not os.path.isfile(file_name):    #i don't think this  is reqd'''

    '''optimizer = tf.train.AdamOptimizer(learning_rate = n).minimize(loss())
            return tflearn.DNN(net)
    '''

    @staticmethod
    def do_action():
        tl = traci._trafficlight.TrafficLightDomain()
        tl.getCompleteRedYellowGreenDefinition()

    def train_network(self, noq_train, learning_rate):
        if os.path.exists("model.tfl"):
            self.model_object.load("model.tfl")
        else:
            self.create_model(learning_rate)
        for _ in noq_train:
            self.store_q_values(self.state_buffer[-1])  # not sure whether self.model_object.fn_name or self.fn_name

        self.model_object.fit(self.state_buffer, self.q_value_buffer)

        self.q_value_buffer = []
        self.state_buffer = []

        self.model_object.save("model.tfl")



'''
a = Network(3,2)
a.create_layer(15,3)
a.create_layer(12,15)
a.create_layer(1,12)
b = a.return_output(np.array([1.,2.,3.]),1,lambda x : tf.square(x))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(b))


print(sess.run(a.weights[0]))
print(sess.run(a.biases[0]).shape)
'''
