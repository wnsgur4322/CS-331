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
#  - bfs (for breadth-first search)
#  - dfs (for depth-first search)
#  - iddfs (for iterative deepening depth-first search)
#  * informed
#  - astar (for A-Star search below)

# command line prompt with argparse library
parser = arg.ArgumentParser()
parser.add_argument('goal_state_file')
parser.add_argument('mode')
parser.add_argument('output_file')
args = parser.parse_args()

if __name__ == "__main__":
    # open file and then get data
    with open(args.goal_state_file, "r") as f_1:
        for i, line in enumerate(f_1):
            if i == 0:
                #first line is left bank (chicken, wolves, boat)
                left_bank = list(map(int, line.strip().split(',')))
            else:
                #second line is right bank (chicken, wolves, boat)
                right_bank = list(map(int, line.strip().split(',')))
 
    print(left_bank)
    print(right_bank)
            
    
