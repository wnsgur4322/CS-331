# CS 331 - Spring 2020
# Programming Assignment 1 - uninformed and informed search
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import math
import argparse as arg
import queue

# code should take the following command line arguments:
# < initial state file > < goal state file > < mode > < output file >

# The mode argument is either:
#  * uninformed
#  - bfs (for breadth-first search)
#  - dfs (for depth-first search)
#  - iddfs (for iterative deepening depth-first search)
#  * informed
#  - astar (for A-Star search below)

# command line prompt with argparse library
parser = arg.ArgumentParser()
parser.add_argument('initial_state_file')
parser.add_argument('goal_state_file')
parser.add_argument('mode')
parser.add_argument('output_file')
args = parser.parse_args()

#set up class Vertex for storing each information
class node:
    def __init__(self):
        self.neighbors = []
        self.state = 'unvisited'
        self.distance = 0

    def add_neighbor(self, node):
        if node not in self.neighbors:
            self.neighbors.append(node)


#set up class Graph for applying algorithms
class Graph:
    def __init__(self):
	    self.nodes = {}
	    self.possible = True
	    self.root = None
	    self.counter = 0

# state class is for the status of left and right banks
class State:
    def __init__(self, chicken_right, wolf_right, chicken_left, wolf_left, boat_location):
        self.chicken_right = chicken_right
        self.chicken_left = chicken_left
        self.wolf_right = wolf_right
        self.wolf_left = wolf_left
        self.boat_location = boat_location

    def __repr__(self):
        return ("Right bank: {0} chicken, {1} wolf \nLeft bank: {2} chicken, {3} wolf \nboat location: {4} (True: Right, False: Left)".format(self.chicken_right, self.wolf_right, self.chicken_left, self.wolf_left, self.boat_location))
     
def find_boat(bank):
    # True for right
    if bank[2] == 1:
        return True
    #False for left
    if bank[5] == 1:
        return False


def successor(num, bank):
    # non-boat bank = no successor
    if bank[2] == 0:
        print("this bank doesn't have a boat for loading animals")
        return
    
    # 1. put one chicken in the boat
    

def bfs():
    #if left_bank == right_bank:
    #    return True

    #frontier =
    return 0

def dfs():
    return 0

def iddfs():
    return 0

def astar():
    return 0

if __name__ == "__main__":
    # open file and then get data
    with open(args.initial_state_file, "r") as f_1:
        for i, line in enumerate(f_1):
            if i == 0:
                #first line is left bank (chicken, wolves, boat)
                init_left_bank = list(map(int, line.strip().split(',')))
            else:
                #second line is right bank (chicken, wolves, boat)
                init_right_bank = list(map(int, line.strip().split(',')))
 
    print(init_left_bank)
    print(init_right_bank)

    init_boat_location = find_boat(list(init_right_bank + init_left_bank))
    print(init_boat_location)

    # open file and then get data
    with open(args.goal_state_file, "r") as f_1:
        for i, line in enumerate(f_1):
            if i == 0:
                #first line is left bank (chicken, wolves, boat)
                goal_left_bank = list(map(int, line.strip().split(',')))
            else:
                #second line is right bank (chicken, wolves, boat)
                goal_right_bank = list(map(int, line.strip().split(',')))
 
    print(goal_left_bank)
    print(goal_right_bank)
    goal_boat_location = find_boat(list(goal_right_bank + goal_left_bank))
    print(goal_boat_location)

    init_state = State(init_right_bank[0],init_right_bank[1],init_left_bank[0],init_left_bank[1],init_boat_location)
    goal_state = State(goal_right_bank[0],goal_right_bank[1],goal_left_bank[0],goal_left_bank[1],goal_boat_location)

    print(init_state)
    print(goal_state)
    if args.mode == "bfs":
        result = bfs()
    elif args.mode == "dfs":
        result = dfs()
    elif args.mode == "iddfs":
        result = iddfs()
    elif args.mode == "astar":
        result = astar()
            
    