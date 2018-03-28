import os, sys
from tools.sumolib import checkBinary
import learner1
import tools.traci as traci
import tensorflow as tf

tf.reset_default_graph()

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = checkBinary("sumo")
sumoCmd = [sumoBinary, "-c", "config1.sumo.cfg"]

traci.start(sumoCmd)

a = traci._defaultDomains[4]
print("\n")
tl = []
for ID in a.getIDList():
    tl.append(learner1.Network(juncID=ID, trafficID=ID))
for t in tl:
    t.pipeline(delta=2000, buffer_size=500, gamma=0.97, learningRate=0.1, training_steps=10, Alpha=0.5,
               future=5)
"""
object1.pipeline(delta=2000, buffer_size=500, gamma=0.97, learningRate=0.1, training_steps=10, Alpha=0.5,
                 future=5)
object2.pipeline(delta=3000, buffer_size=500, gamma=0.98, learningRate=0.05, training_steps=10, Alpha=0.6,
                 future=4)
object3.pipeline(delta=3000, buffer_size=500, gamma=0.98, learningRate=0.05, training_steps=10, Alpha=0.6,
                 future=4)
object4.pipeline(delta=3000, buffer_size=500, gamma=0.98, learningRate=0.05, training_steps=10, Alpha=0.6,
                 future=4)
object5.pipeline(delta=3000, buffer_size=500, gamma=0.98, learningRate=0.05, training_steps=10, Alpha=0.6,
                 future=4)
object6.pipeline(delta=3000, buffer_size=500, gamma=0.98, learningRate=0.05, training_steps=10, Alpha=0.6,
                 future=4)
object7.pipeline(delta=3000, buffer_size=500, gamma=0.98, learningRate=0.05, training_steps=10, Alpha=0.6,
                 future=4)
object8.pipeline(delta=3000, buffer_size=500, gamma=0.98, learningRate=0.05, training_steps=10, Alpha=0.6,
                 future=4)
"""
traci.close(False)
