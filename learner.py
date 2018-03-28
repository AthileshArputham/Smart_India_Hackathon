import tensorflow as tf
import tools.traci as traci
import tools.sumolib as sumolib
import learner1


class Learner:

    def __init__(self, no_of_lanes, state, junctionID, trafficID):
        self.nol = no_of_lanes
        self.state = state
        self.jID = junctionID
        self.trafficID = trafficID
        self.net_file = sumolib.net.readNet("map1.net.xml")
        self.neural_network = None


    def create_network(self, no_of_layers):  # take inputs
        self.neural_network = learner1.Network(self.nol, 3)



