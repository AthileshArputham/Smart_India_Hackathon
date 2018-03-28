import traci


import os, sys
 if 'C:\Program Files (x86)\DLR\Sumo' in os.environ:
     tools = os.path.join(os.environ['C:\Program Files (x86)\DLR\Sumo'], 'tools')
     sys.path.append(tools)
 else:   
     sys.exit("please declare environment variable 'C:\Program Files (x86)\DLR\Sumo'")

sumoBinary = "C:\Program Files (x86)\DLR\Sumo\bin"
sumoCmd = [sumoBinary, "-c", "test1.sumo.cfg"]

traci.start(sumoCmd)

