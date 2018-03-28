import os

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from scipy import stats

import MapNetwork
import tools.sumolib as sumolib
import tools.traci as traci

plt.style.use("fivethirtyeight")


class Network:

    def __init__(self, juncID, trafficID):

        self.juncID = juncID
        self.state_buffer = []  # stores the states
        self.q_value_buffer = []  # stores the Q(s,a)
        self.action_index_buffer = []
        self.reward_buffer = []

        self.trafficID = trafficID
        self.detectors = MapNetwork.e2forJunc(self.trafficID)
        self.nol = int(len(self.detectors))
        self.model_object = None
        self.net_file = sumolib.net.readNet("vjw.net.xml")
        self.no_edges = len(self.net_file.getNode(self.juncID)._incoming)
        self.model_file = str(self.juncID) + "_" + str(self.trafficID) + ".json"
        self.weights_file = str(self.juncID) + "_" + str(self.trafficID) + ".h5"
        self.negative_index = []
        self.history = None
        # self.inputs = tf.placeholder(tf.float32, shape=[1, 2 * self.no_edges + self.nol])
        # self.f = open(str(self.juncID) + "_" + str(self.trafficID) + ".txt", "w+")
        print("No of Incoming Lanes for The Object", self.trafficID, ":", self.nol, "\n", "No of Edges:", self.no_edges)

    def get_state(self):
        t1 = traci._defaultDomains[4]
        t1 = t1.getCompleteRedYellowGreenDefinition(self.trafficID)
        t2 = t1[0].return_phase()
        return t2

    def get_loads(self):

        x = traci._defaultDomains[4]
        b = x.getControlledLanes(self.trafficID)
        b1 = []
        loads = []
        a = traci._defaultDomains[6]
        for i in b:
            if i not in b1:
                loads.append(a.getLastStepVehicleNumber(i))
                b1.append(i)
        return loads, len(loads)  # unpack tuple

    def create_inputs(self, state=None):
        if state is None:
            a = self.get_state()
        else:
            a = list(state)
        (b, c) = self.get_loads()
        _input = a + b  # c is removed
        return _input

    def get_reward(self):
        reward = 0
        e2 = traci._defaultDomains[1]

        for detector in list(set(self.detectors)):
            reward -= (e2.getLastStepHaltingNumber(detector))

        self.reward_buffer.append(reward)

    def updateQTable(self, disco_factor, alpha, k=5):  # we need to write reward fn.#added loss function

        def QmaxFinder(i):
            temp = self.q_value_buffer[i + 1]
            return np.max(temp.reshape(2 * self.no_edges - 1, 3), axis=1)

        last_index = len(self.q_value_buffer) - 1 - k

        reward = self.reward_buffer[last_index + k]
        init_input = self.state_buffer[0][0][:2 * self.no_edges]
        last_input = self.state_buffer[last_index][0][:2 * self.no_edges]
        for i in range(len(init_input)):
            reward += stats.norm(loc=init_input[i], scale=0.01).pdf(last_input[i])
        reward -= self.negative_index[last_index] * 5
        q = self.q_value_buffer[last_index][0][self.action_index_buffer[last_index - 1]]

        self.q_value_buffer[last_index][0][self.action_index_buffer[last_index - 1]] += \
            alpha * (reward - q)

        while last_index >= 0:
            last_index -= 1
            q = self.q_value_buffer[last_index][0][self.action_index_buffer[last_index - 1]]
            reward = self.reward_buffer[last_index + k]
            Qmax = QmaxFinder(last_index)
            init_input = self.state_buffer[0][0][:2 * self.no_edges]
            last_input = self.state_buffer[last_index][0][:2 * self.no_edges]
            for i in range(len(init_input)):
                reward += stats.norm(loc=init_input[i], scale=1).pdf(last_input[i])
            reward -= self.negative_index[last_index]
            self.q_value_buffer[last_index][0][self.action_index_buffer[last_index - 1]] += alpha * (
                    reward + disco_factor * Qmax - q)

    def create_model(self, n):

        # net = tflearn.input_data(placeholder=self.inputs)
        model = Sequential()
        model.add(Dense(units=64, input_dim=2 * self.no_edges + self.nol, activation='tanh', name='h1',
                        kernel_initializer='random_uniform'))
        model.add(Dense(units=32, activation='tanh', name='h2', kernel_initializer='random_uniform'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=32, activation='tanh', name='h3', kernel_initializer='random_uniform'))
        model.add(Dense(units=6 * self.no_edges - 3, activation=None, name='h4', kernel_initializer='random_uniform'))
        model.compile(optimizer='adam', loss="mean_squared_error")
        self.model_object = model

    def store_q_values(self, input_state):
        if self.model_object is not None:
            #  input_state[:2*self.no_edges] = input_state[:self.no_edges]/9000.
            predict_q = self.model_object.predict(input_state)

            def check(arr):
                a = arr[:2 * self.no_edges]
                count = 0
                if np.any(a) > 80000:
                    count = 1
                elif np.any(a) < 2000:
                    count = 1
                return count

            self.state_buffer.append(input_state)
            self.negative_index.append(check(input_state))
            self.q_value_buffer.append(predict_q)
        # changed the defn.#it is actually get Q values..??not giving actions

    def get_action_index(self, eps):
        array = np.array(self.q_value_buffer[-1])
        rows = 2 * self.no_edges - 1
        action_index_list = []
        array = array.reshape([rows, 3])
        for i in range(rows):
            if np.random.uniform(0, 1) > eps:
                action_index_list.append(np.argmax(array[i]))  # selecting according to maxQvalue
            else:
                action_index_list.append(np.random.choice([0, 1, 2]))

        # selecting random action

        required_q_index = np.argmax(array, axis=1) - 1
        b = list(required_q_index + 1)
        self.action_index_buffer.append(b)
        return b, np.max(array, axis=1)  # changed the return statement

    def next_state(self, curr_state, delta):
        action = self.action_index_buffer[-1]  # take care of gradual decrease of epsilon
        action = np.array(action)
        last_action_index = -np.sum(action - 1) + 1
        action = np.append(action, last_action_index)
        action = (action - 1) / np.max(np.abs(action))
        action = action * delta
        cur_ = curr_state[0][:2 * self.no_edges]
        state_next = np.array(cur_) + action
        t1 = traci._defaultDomains[4]
        t2 = t1.getCompleteRedYellowGreenDefinition(self.trafficID)[0]
        t2.set_stateDuration(state_next.astype(int))
        t1.setCompleteRedYellowGreenDefinition(self.trafficID, t2)

        # setting next state in sumo
        return state_next

    '''    def train(self):
            file_name = "variables.tflearn"
            if not os.path.isfile(file_name):    #i don't think this  is reqd'''

    '''optimizer = tf.train.AdamOptimizer(learning_rate = n).minimize(loss())
            return tflearn.DNN(net)
    '''

    def save_model(self):
        # serialize model to JSON
        model_json = self.model_object.to_json()
        with open(self.model_file, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model_object.save_weights(self.weights_file)

    def load_model(self):
        # load json and create model
        json_file = open(self.model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_object = model_from_json(loaded_model_json)
        # load weights into new model
        self.model_object.load_weights(self.weights_file)

    def train_network(self, learning_rate):
        if os.path.exists(self.model_file) and os.path.exists(self.weights_file):
            self.load_model()
            self.model_object.compile(optimizer='adam', loss='mean_squared_error')
        else:

            self.create_model(learning_rate)
        '''for _ in noq_train:
            self.store_q_values(self.state_buffer[-1])  # not sure whether self.model_object.fn_name or self.fn_name
        '''

        # self.f.write("State Buffer: "+str(self.state_buffer1[0][])+"\n")
        # self.f.write("Q Buffer:" + str(self.q_value_buffer)+ "\n")
        # self.f.close()
        # for a in self.state_buffer:
        #    a = a.resize(15)
        # self.model_object.fit(self.state_buffer, self.q_value_buffer)
        a = np.array(self.state_buffer)
        b = np.array(self.q_value_buffer)
        self.state_buffer = a.reshape((a.shape[0], a.shape[2]))
        self.q_value_buffer = b.reshape((b.shape[0], b.shape[2]))
        #        self.state_buffer[:,:2*self.no_edges] = self.state_buffer[:,:2*self.no_edges]/9000.
        self.history = self.model_object.fit(self.state_buffer, self.q_value_buffer, epochs=50)
        self.q_value_buffer = []
        self.state_buffer = []
        self.save_model()

    def pipeline(self, delta, buffer_size, gamma, Alpha, learningRate, training_steps, future):
        # transposed
        self.create_model(learningRate)
        j = 0
        while j < training_steps:
            i = 0
            eps = np.exp(-j)
            # plot_model(self.model_object,to_file = 'model.png')
            next_state = None
            while i < buffer_size:
                traci.simulationStep()
                input_state = np.array(self.create_inputs(next_state)).reshape([1, 2 * self.no_edges + self.nol])
                self.store_q_values(input_state)
                self.get_action_index(eps)
                next_state = self.next_state(input_state, delta)
                self.get_reward()
                # self.f.write(
                #    str(i) + " : " + str(traci._defaultDomains[4].getCompleteRedYellowGreenDefinition(self.trafficID)))
                i += 1
            self.updateQTable(gamma, Alpha, future)
            self.train_network(learningRate)
            print("_________________________________________________________________")
            plt.figure(figsize=(15,15))
            plt.plot(self.history.history['loss'])
            plt.title('Model loss' + "for ID:" + str(self.juncID) + " in training step:" + str(j))
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(str(self.juncID)+" " + str(training_steps)+".jpg")
            j += 1
