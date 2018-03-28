from tools.sumolib.net.node import Node as nd
import numpy as np

class node(nd):
    def __init__(self,ID,TYPE,COORD,INCLANES):
        self.Node_object = nd(id=ID,type=TYPE,coord=COORD,incLanes=INCLANES)
        self.parent = None
        self.g = 0
        self.f = 0


    def get_children(graph,node):


def f_cost(point1, point2):
    return (point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2

def printReversedList(templist):
    for i in reversed(templist):
        print(i)

def aStarImplementation(source,target,graph):
    openSet = set()
    closedSet = set()
    currentNode = source
    openSet.add(currentNode)
    while openSet:
        currentNode =  #choose node either at random or min

        if currentNode == target :
            pathGenerated = []
            while currentNode.parent:
                pathGenerated.append(currentNode)
                currentNode = currentNode.parent
            pathGenerated.append(currentNode)
            printReversedList(pathGenerated)  #write path somewhere dont returna anything
        openSet.remove(currentNode)
        closedSet.add(currentNode)

        if node in get_children(currentNode,graph):
            if node in closedSet:
                continue
            if node in openSet:
                f_value = currentNode.f + f_cost(node.Node_object,currentNode)
                if f_value < node.f:
                    node.f = f_value
                    node.parent = currentNode
                    openSet.add(node)
