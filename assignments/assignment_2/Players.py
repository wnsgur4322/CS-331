'''
    Erich Kramer - April 2017
    Apache License
    If using this code please cite creator.

'''
# CS 331 - Spring 2020
# Programming Assignment 2 - Simplified Othello
# Junhyeok Jeong, jeongju@oregonstate.edu

import OthelloBoard
import math

class Player:
    def __init__(self, symbol):
        self.symbol = symbol

    #PYTHON: use obj.symbol instead
    def get_symbol(self):
        return self.symbol
    
    #parent get_move should not be called
    def get_move(self, board):
        raise NotImplementedError()



class HumanPlayer(Player):
    def __init__(self, symbol):
        Player.__init__(self, symbol)

    def clone(self):
        return HumanPlayer(self.symbol)
        
#PYTHON: return tuple instead of change reference as in C++
    def get_move(self, board):
        succ_move = self.successor(board, self.symbol)
        print("available moves: {0}".format(succ_move))
        col = int(input("Enter col:"))
        row = int(input("Enter row:"))
        return  (col, row)

    # I added successor function for human player too because of convenience
    def successor(self, board, symbol):
        # list of successors
        succ = []
        
        # check all spots are available to put on the board
        for c in range(4):
            for r in range(4):
                if board.is_legal_move(c, r, symbol) == True:
                    succ.append((c, r))
        
        return succ


class MinimaxPlayer(Player):

    def __init__(self, symbol):
        Player.__init__(self, symbol)
        if symbol == 'X':
            self.oppSym = 'O'
        else:
            self.oppSym = 'X'
    
    # to calculate the utility score for minimax
    def get_utility(self, board):
        return board.count_score(self.symbol) - board.count_score(self.oppSym)

    def get_move(self, board):
        print("AI: CALCULATING ...")
        max_val = -(math.inf)   # express for negative infinity
        min_val = math.inf      # express for infinity
        utility = 0         # value for choosing best successor

        succ_move = self.successor(board, self.symbol)
        print("available moves: {0}".format(succ_move))

        for i in range(len(succ_move)):
            # set up virtual board before calculate utility
            virtual_board = board.cloneOBoard()
            virtual_board.play_move(succ_move[i][0], succ_move[i][1], self.symbol)
    
            # if AI is first player, get max of min value
            if self.symbol == board.p1_symbol:
                utility = self.min_value(virtual_board)
                if utility >= max_val:
                    max_val = utility
                    col = succ_move[i][0]
                    row = succ_move[i][1]
            
            # if AI is the seconnd player, get min of max value
            else:
                utility = self.min_value(virtual_board)
                if utility >= max_val:
                    max_val = utility
                    col = succ_move[i][0]
                    row = succ_move[i][1]

        return (col, row)

    
    def max_value(self, board):
        # v <= negative infinity
        max_val = -(math.inf)
        utility = 0
        # if there is no more reamining spot on the board, then return utility directly
        if board.has_legal_moves_remaining(self.symbol) == False and board.has_legal_moves_remaining(self.oppSym) == False:
            return self.get_utility(board)
        
        # get available successors
        succ_move = self.successor(board, self.symbol)
        if len(succ_move) == 0:
            utility = self.min_value(board)
            if utility >= max_val:
                max_val = utility
        else:
            for i in range(len(succ_move)):
                # set up virtual board before calculate utility for recording
                virtual_board = board.cloneOBoard()
                virtual_board.play_move(succ_move[i][0], succ_move[i][1], self.symbol)
                # get utility value for each virtual move
                utility = self.min_value(virtual_board)
                if utility >= max_val:
                    max_val = utility
        
        return max_val

    def min_value(self, board):
        # v <= infinity
        min_val = math.inf
        utility = 0
        # if there is no more reamining spot on the board, then return utility directly
        if board.has_legal_moves_remaining(self.symbol) == False and board.has_legal_moves_remaining(self.oppSym) == False:
            return self.get_utility(board)
        
        #get available successors
        succ_move = self.successor(board, self.oppSym)
        if len(succ_move) == 0:
            utility = self.max_value(board)
            if utility < min_val:
                min_val = utility
        else:
            for i in range(len(succ_move)):
                # set up virtual board before calculate utility for recording
                virtual_board = board.cloneOBoard()
                virtual_board.play_move(succ_move[i][0], succ_move[i][1], self.oppSym)
                # get utility value for each virtual move
                utility = self.max_value(virtual_board)
                if utility < min_val:
                    min_val = utility
        return min_val

    def successor(self, board, symbol):
        # list of successors
        succ = []
        
        # check all spots are available to put on the board
        for c in range(4):
            for r in range(4):
                if board.is_legal_move(c, r, symbol) == True:
                    succ.append((c, r))
        
        return succ
        





