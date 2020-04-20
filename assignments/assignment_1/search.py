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
        #print("Node.__eq__ is called")
        if isinstance(other, Node):
            return self.depth == other.depth
        else:
            return False

    def __hash__(self):
        #print("Node.__hash__ is called")
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
        #print("State.__eq__ is called")
        return self.__dict__ == other.__dict__

    def __hash__(self):
        #print("State.__hash__ is called")
        return hash(tuple(sorted(self.__dict__.items())))

def find_boat(bank):
    # True for right
    if bank[2] == 1:
        return True
    #False for left
    if bank[5] == 1:
        return False

def valid_check(State):
    if State.chicken_left >= 0 and State.chicken_right >= 0 and State.wolf_left >= 0 and State.wolf_right >= 0:
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
    elif state.boat_location == False:  
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

        # add initial node at explored list
        explored.add(cur_node)
        # create a set of successor for the puzzle
        succs = successor(cur_node)

        for succ_node in succs:
            num_expanded += 1
            # if the element node of the set of successor is not included in explored list, then compare the node with goal state
            if succ_node not in explored:
                # if succ_node state condition has the same with goal state then done !
                if succ_node.state.chicken_left == goal_state.chicken_left and succ_node.state.wolf_left == goal_state.wolf_left and succ_node.state.boat_location == False:
                    return succ_node, num_expanded
                # add new succesor node on frontier
                frontier.append(succ_node)
    
    return False, num_expanded

def dfs(init_state, goal_state):
    #initialize the frontier and input a node of the initial state
    frontier = []
    frontier.append(Node(None, init_state, None))

    #initialize the explored set to be empty
    explored = set()
    num_expanded = 0

    while frontier :
        #Last in first out queue as DFS
        cur_node = frontier[len(frontier) - 1]
        frontier.pop()

        # add initial node at explored list
        explored.add(cur_node)
        # create a set of successor for the puzzle
        succs = successor(cur_node)
        print(len(succs))
        
        for succ_node in succs:
            num_expanded += 1
            # if the element node of the set of successor is not included in explored list, then compare the node with goal state
            if succ_node not in explored:
                # if succ_node state condition has the same with goal state then done !
                if succ_node.state.chicken_left == goal_state.chicken_left and succ_node.state.wolf_left == goal_state.wolf_left and succ_node.state.boat_location == False:
                    return succ_node, num_expanded
                # add new succesor node on frontier
                frontier.append(succ_node)
    
    return False, num_expanded

# Page 88~89 - Iterative Deepening Depth-First Search
def recursive_dls(cur_node, goal_state, max_depth, num_expanded):
    if cur_node.state.chicken_left == goal_state.chicken_left and cur_node.state.wolf_left == goal_state.wolf_left and cur_node.state.boat_location == False:
        return cur_node, num_expanded
    elif max_depth == 0:
        return "cutoff", num_expanded
    else:
        cutoff_occurred = False
        succs = successor(cur_node)
        for succ_node in succs:
            num_expanded += 1
            result, num_expanded = recursive_dls(succ_node, goal_state, max_depth - 1, num_expanded)
            if result == "cutoff":
                cutoff_occurred = True
            elif result:
                # Done!
                return result, num_expanded
        if cutoff_occurred == True:
            return "cutoff", num_expanded
        else:
            return False, num_expanded

def dls(init_state, goal_state, max_depth, num_expanded):
    init_node = Node(None, init_state, None)
    return recursive_dls(init_node, goal_state, max_depth, num_expanded)

def iddfs(init_state, goal_state):
    num_expanded = 0
    max_depth = 0
    limit_depth = 15
    while True:
        result, num_expanded = dls(init_state, goal_state, max_depth, num_expanded)
        if result != "cutoff":
            return result, num_expanded
        else:
            print("Unreached goal state with %d maximum depth" % max_depth)
        max_depth += 1
        num_expanded = 0

        if max_depth == limit_depth:
            return False, num_expanded

import queue as qu
# Page 99 - A* Search (Recursive Best-First Search)
def astar(init_state, goal_state):
    if(init_state == goal_state):
        return Node(None, init_state, None), 0

    frontier = qu.PriorityQueue()
    frontier.put((0, Node(None, init_state, None)))


    explored = set()
    num_expanded = 0

    while frontier:
        cur_node = frontier.get()
        explored.add(cur_node[1])
        succs = successor(cur_node[1])

        for succ_node in succs:
            if succ_node in explored:
                num_expanded += 1
            if succ_node not in explored:
                num_expanded += 1
                if succ_node.state.chicken_left == goal_state.chicken_left and succ_node.state.wolf_left == goal_state.wolf_left:
                    return succ_node, num_expanded
                frontier.put(((succ_node.state.chicken_right + succ_node.state.wolf_right), succ_node))


def print_path(path):
    if path == False:
        print("The %s algorithm can't find solution path" % args.mode)
        return 0
    path_statements = []
    solution_path = ""

    while path != None:
        path_statements.append(path.successor_statement)
        path = path.parent_node

    for statement in path_statements:
        if statement != None:
            solution_path += statement + "\n"
    
    print("total steps: %d" % (len(path_statements) - 1))
    return solution_path

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
        path, num_expanded = dfs(init_state, goal_state)
    elif args.mode == "iddfs":
        path, num_expanded = iddfs(init_state, goal_state)
    elif args.mode == "astar":
        path, num_expanded = astar(init_state, goal_state)

    print("\n-- your mode is %s --" % args.mode)
    
    # if the algorithm can find solution path, then print out and write result.txt
    if path != False:
        print("the %s algorithm expended %d nodes" % (args.mode, num_expanded))
        print("-- solution path --")
        print(print_path(path))

        with open(args.output_file, "w") as f_3:
            f_3.write("-- your mode is %s --\n" % args.mode)
            f_3.write("the %s algorithm expended %d nodes\n" % (args.mode, num_expanded))
            f_3.write("-- solution path --\n")
            f_3.write(print_path(path))
    
    else:
        print("The %s algorithm can't find solution path" % args.mode)
