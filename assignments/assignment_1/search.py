# CS 331 - Spring 2020
# Programming Assignment 1 - uninformed and informed search
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import math
import argparse as arg

# code should take the following command line arguments:
# < initial state file > < goal state file > < mode > < output file >

# The mode argument is either:
#  * uninformed
#  - bfs (for breadth-first search, using queue FIFO)
#  - dfs (for depth-first search, using stack LIFO)
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

#set up the Node class to store frontier and explored queue (list)
class Node:
    def __init__(self, parent_node, State, successor_statement):
        self.parent_node = parent_node
        self.state = State
        self.depth = 0
        self.successor_statement = successor_statement
    
    def __repr__(self):
        return str(self.state)

    def __eq__(self, other):
        print("Node.__eq__ is called")
        if isinstance(other, Node):
            return self.depth == other.depth
        else:
            return False

    def __hash__(self):
        print("Node.__hash__ is called")
        return self.state.__hash__()


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

    def __eq__(self, other):
        print("State.__eq__ is called")
        return self.__dict__ == other.__dict__

    def __hash__(self):
        print("State.__hash__ is called")
        return hash(tuple(sorted(self.__dict__.items())))

def find_boat(bank):
    # True for right
    if bank[2] == 1:
        return True
    #False for left
    if bank[5] == 1:
        return False

def valid_check(State):
    if State.chicken_left >= 0 and State.chicken_right >= 0 and State.wolf_left >= 0 and State.chicken_right >= 0:
        if (State.chicken_right >= State.wolf_right or State.chicken_right == 0) and (State.chicken_left >= State.wolf_left or State.chicken_left == 0):
            return True
    else:
        return False 


def successor(parent_node):
    # set state to check boat location and create successor set to store
    state = parent_node.state
    successors = set()

    # if boat is on right bank
    if state.boat_location == True:
    
        # 1. put one chicken in the boat
        r_succ_1 = State(state.chicken_right - 1, state.wolf_right, state.chicken_left + 1, state.wolf_left, False)
        
        # check the successor is valid
        if valid_check(r_succ_1) == True:
            successors.add(Node(parent_node, r_succ_1, "put one chicken in the boat"))
        
        # 2. put two chicken in the boat
        r_succ_2 = State(state.chicken_right - 2, state.wolf_right, state.chicken_left + 2, state.wolf_left, False)
        
        # check the successor is valid
        if valid_check(r_succ_2) == True:
            successors.add(Node(parent_node, r_succ_2, "put two chicken in the boat"))

        # 3. put one wolf in the boat
        r_succ_3 = State(state.chicken_right, state.wolf_right - 1, state.chicken_left, state.wolf_left + 1, False)
        
        # check the successor is valid
        if valid_check(r_succ_3) == True:
            successors.add(Node(parent_node, r_succ_3, "put one wolf in the boat"))

        # 4. put one wolf and one chicken in the boat
        r_succ_4 = State(state.chicken_right - 1, state.wolf_right - 1, state.chicken_left + 1, state.wolf_left + 1, False)
        
        # check the successor is valid
        if valid_check(r_succ_4) == True:
            successors.add(Node(parent_node, r_succ_4, "put one wolf and one chicken in the boat"))

        # 5. put two wolves in the boat
        r_succ_5 = State(state.chicken_right, state.wolf_right - 2, state.chicken_left, state.wolf_left + 2, False)
        
        # check the successor is valid
        if valid_check(r_succ_5) == True:
            successors.add(Node(parent_node, r_succ_5, "put two wolves in the boat"))

    # if boat is on left bank
    else:  
        # 1. put one chicken in the boat
        l_succ_1 = State(state.chicken_right + 1, state.wolf_right, state.chicken_left - 1, state.wolf_left, True)
        
        # check the successor is valid
        if valid_check(l_succ_1) == True:
            successors.add(Node(parent_node, l_succ_1, "put one chicken in the boat"))
        
        # 2. put two chicken in the boat
        l_succ_2 = State(state.chicken_right + 2, state.wolf_right, state.chicken_left - 2, state.wolf_left, True)
        
        # check the successor is valid
        if valid_check(l_succ_2) == True:
            successors.add(Node(parent_node, l_succ_2, "put two chicken in the boat"))

        # 3. put one wolf in the boat
        l_succ_3 = State(state.chicken_right, state.wolf_right + 1, state.chicken_left, state.wolf_left - 1, True)
        
        # check the successor is valid
        if valid_check(l_succ_3) == True:
            successors.add(Node(parent_node, l_succ_3, "put one wolf in the boat"))

        # 4. put one wolf and one chicken in the boat
        l_succ_4 = State(state.chicken_right + 1, state.wolf_right + 1, state.chicken_left - 1, state.wolf_left - 1, True)
        
        # check the successor is valid
        if valid_check(l_succ_4) == True:
            successors.add(Node(parent_node, l_succ_4, "put one wolf and one chicken in the boat"))

        # 5. put two wolves in the boat
        l_succ_5 = State(state.chicken_right, state.wolf_right + 2, state.chicken_left, state.wolf_left - 2, True)
        
        # check the successor is valid
        if valid_check(l_succ_5) == True:
            successors.add(Node(parent_node, l_succ_5, "put two wolves in the boat"))    

    return successors

def bfs(init_state, goal_state):
    #initialize the frontier using the initial state of the problem (using python list)
    frontier = []
    frontier.append(Node(None, init_state, None))

    #initialize the explored set to be empty
    explored = set()
    num_expanded = 0

    while len(frontier) != 0:
        #first in first out queue as BFS
        cur_node = frontier[0]
        frontier.pop(0)

        explored.add(cur_node)
        succs = successor(cur_node)
        print(len(succs))
        for succ_node in succs:
            if succ_node in explored:
                num_expanded += 1
                print(num_expanded)
            if succ_node not in explored:
                # if succ_node state condition has the same with goal state then done !
                if succ_node.state.chicken_left == goal_state.chicken_left and succ_node.state.wolf_left == goal_state.wolf_left:
                    return succ_node, num_expanded
                frontier.append(succ_node)


def dfs():
    return 0

# Page 88~89 - Iterative Deepening Depth-First Search
def dls(problem, limit):
    return 0

def rdls(node, problem, limit):
    if problem == goal:
        return solution(node)
    elif limit == 0:
        return cutoff()
    else:
        is_cutoff = False
    #    for i in range(0, limit):
            #do recursive dls
            # in this for or while 
    return 0

def iddfs(problem):
    depth = 0
    infi = math.inf
    for i in range(0, infi):
        if(dls(problem, depth)):
            return True
    return False

# Page 99 - A* Search (Recursive Best-First Search)
def astar():
    return 0

def action_sequence(node):
    actions = []
    while node != None:
        actions.append(node.successor_statement)
        node = node.parent_node
    return list(reversed(actions))

def print_path(path):
    result = ""
    path_statements = []

    while path != None:
        path_statements.append(path.successor_statement)
        path = path.parent_node

    for statement in path_statements:
        if statement != None:
            print(statement)
    
    print("total steps: %d" % (len(path_statements) - 1))

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

    init_boat_location = find_boat(list(init_right_bank + init_left_bank))

    # open file and then get data
    with open(args.goal_state_file, "r") as f_2:
        for i, line in enumerate(f_2):
            if i == 0:
                #first line is left bank (chicken, wolves, boat)
                goal_left_bank = list(map(int, line.strip().split(',')))
            else:
                #second line is right bank (chicken, wolves, boat)
                goal_right_bank = list(map(int, line.strip().split(',')))

    goal_boat_location = find_boat(list(goal_right_bank + goal_left_bank))

    # print out initial and goal state and check validation
    init_state = State(init_right_bank[0],init_right_bank[1],init_left_bank[0],init_left_bank[1],init_boat_location)
    goal_state = State(goal_right_bank[0],goal_right_bank[1],goal_left_bank[0],goal_left_bank[1],goal_boat_location)

    print(init_state)
    print(goal_state)

    print("input files validation check: initial state: %s, goal state: %s" % (valid_check(init_state), valid_check(goal_state)))

    # apply algorithms depending on the user's mode input
    if args.mode == "bfs":
        path, num_expanded = bfs(init_state, goal_state)
    elif args.mode == "dfs":
        path, num_expanded = dfs()
    elif args.mode == "iddfs":
        result = iddfs()
    elif args.mode == "astar":
        result = astar()

    print("\n-- your mode is %s --" % args.mode)
    print("the %s algorithm expened %d nodes" % (args.mode, num_expanded))
    print("-- solution path --")
    print_path(path)