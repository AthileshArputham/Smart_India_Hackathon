import tools.traci as traci
import tools.sumolib as sumolib

net_file = sumolib.net.readNet("map1.net.xml")
tl = traci._lanearea.LaneAreaDomain()


def create_dict():
    dictionary = {}
    for e2 in tl.getIDList():
        dictionary[e2] = tl.getLaneID(e2)
    dictionary = dict([[v, k] for k, v in dictionary.items()])
    return dictionary


print(traci._connections)
tr = traci._trafficlight.TrafficLightDomain(name="trafficlight")

def e2forJunc(TrafficID):
    inc_lanes = tr.getControlledLanes(TrafficID)
    detect_lanes = []
    a = create_dict()
    for lane in inc_lanes:
        detect_lanes.append(a[lane])
    detect_lanes = list(set(detect_lanes))
    return detect_lanes
