import os, sys
import traci

from sumolib import checkBinary
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
sumoBinary = "C:\\Program Files (x86)\\DLR\\Sumo\\bin\\sumo-gui.exe"
sumoBinary = checkBinary("sumo")
sumoCmd = [sumoBinary, "-c", "config1.sumo.cfg"]

traci.start(sumoCmd)

tld=traci.trafficlight.getCompleteRedYellowGreenDefinition("gneJ58")
tld1=traci.trafficlight.getRedYellowGreenState("gneJ58")
tld2=traci.trafficlight.getPhaseDuration("gneJ58")
print(tld2)
traci.close()
#obj._readlogics(result)