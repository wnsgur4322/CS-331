'''
    Erich Kramer - April 2017
    Apache License
    If using this code please cite creator.

'''
# CS 331 - Spring 2020
# Programming Assignment 2 - Simplified Othello
# Junhyeok Jeong, jeongju@oregonstate.edu

from Players import *
import sys
import OthelloBoard


class GameDriver:
    def __init__(self, p1type, p2type, num_rows, num_cols):
        if p1type.lower() in "human":
            self.p1 = HumanPlayer('X')

        elif p1type.lower() in "minimax" or p1type in "ai":
            self.p1 = MinimaxPlayer('X')

        else:
            print("Invalid player 1 type!")
            exit(-1)

        if p2type.lower() in "human":
            self.p2 = HumanPlayer('O')

        elif p2type.lower() in "minimax" or p2type in "ai":
            self.p2 = MinimaxPlayer('O')

        else:
            print("Invalid player 2 type!")
            exit(-1)

        self.board = OthelloBoard.OthelloBoard(num_rows, num_cols, self.p1.symbol, self.p2.symbol)
        self.board.initialize()

    def display(self):
        print("Player 1 ( %s ) score: %d\n Player 2 ( %s ) score: %d" % (self.p1.symbol,
                self.board.count_score(self.p1.symbol), self.p2.symbol, self.board.count_score(self.p2.symbol)))
        # fun stuff
        if (sys.argv[1] == "ai" or "minimax") and sys.argv[2] == "human":
            if self.board.count_score(self.p1.symbol) > self.board.count_score(self.p2.symbol):
                print("\nAI: YOU SHOULD PRACTICE MORE, HUMAN\n")
            if self.board.count_score(self.p1.symbol) == self.board.count_score(self.p2.symbol):
                print("\nAI: Ho-oh that's pretty good, HUMAN\n")
            if self.board.count_score(self.p1.symbol) < self.board.count_score(self.p2.symbol):
                print("\nAI: HOW DARE!, HUMAN\n")
        
        if (sys.argv[2] == "ai" or "minimax") and sys.argv[1] == "human":
            if self.board.count_score(self.p2.symbol) > self.board.count_score(self.p1.symbol):
                print("\nAI: YOU SHOULD PRACTICE MORE, HUMAN\n")
            if self.board.count_score(self.p2.symbol) == self.board.count_score(self.p1.symbol):
                print("\nAI: Ho-oh that's pretty good, HUMAN\n")
            if self.board.count_score(self.p2.symbol) < self.board.count_score(self.p1.symbol):
                print("\nAI: HOW DARE!, HUMAN\n")

    # to record player's input path
    def record_path(self, user_input, record):
        record.append(user_input)

        return record

    def process_move(self, curr_player, opponent):
        invalid_move = True
        while(invalid_move):
            (col, row) = curr_player.get_move(self.board)
            if( not self.board.is_legal_move(col, row, curr_player.symbol)):
                print("Invalid move")
            else:
                print("Move:", [col,row], "\n")
                self.board.play_move(col,row,curr_player.symbol)
                return (col, row)


    def run(self, game):
        current = self.p1
        opponent = self.p2
        self.board.display()

        cant_move_counter, toggle = 0, 0

        #main execution of game
        print("Player 1(", self.p1.symbol, ") move:")

        p1_path = []
        p2_path = []
        while True:
            
            if self.board.has_legal_moves_remaining(current.symbol):
                cant_move_counter = 0
                user_input = self.process_move(current, opponent)
                if current.symbol == 'X':
                    self.record_path(user_input, p1_path)
                else:
                    self.record_path(user_input, p2_path)    

                self.board.display()
                game.display()
            else:
                print("Can't move")
                if(cant_move_counter == 1):
                    break
                else:
                    cant_move_counter +=1
            toggle = (toggle + 1) % 2
            if toggle == 0:
                current, opponent = self.p1, self.p2
                print("Player 1(", self.p1.symbol, ") move:")
            else:
                current, opponent = self.p2, self.p1
                print("Player 2(", self.p2.symbol, ") move:")
            

        #decide win/lose/tie state
        state = self.board.count_score(self.p1.symbol) - self.board.count_score(self.p2.symbol)
        if( state == 0):
            print("---------------")
            print("Tie game!!")
            print("\nplayer 1's inputs: {0}".format(p1_path))
            print("player 2's inputs: {0}".format(p2_path))
            # fun stuff
            if ((sys.argv[1] == "ai" or "minimax") and sys.argv[2] == "human") or ((sys.argv[2] == "ai" or "minimax") and sys.argv[1] == "human"):
                print("\nAI: I WAS CAUGHT OFF GUARD, HUMAN")
        elif state >0:
            print("---------------")
            print("Player 1 Wins!")
            print("\nplayer 1's inputs: {0}".format(p1_path))
            print("player 2's inputs: {0}".format(p2_path))
            #fun stuff
            if (sys.argv[1] == "ai" or "minimax") and sys.argv[2] == "human":
                print("\nAI: PATHETIC, HUMAN")
        else:
            print("---------------")
            print("Player 2 Wins!")
            print("\nplayer 1's inputs: {0}".format(p1_path))
            print("player 2's inputs: {0}".format(p2_path))
            #fun stuff
            if (sys.argv[2] == "ai" or "minimax") and sys.argv[1] == "human":
                print("\nAI: PATHETIC, HUMAN")
            


def main():
    if(len(sys.argv)) != 3:
        print("Usage: python3 GameDriver.py <player1 type> <player2 type>")
        exit(1)
    game = GameDriver(sys.argv[1], sys.argv[2], 4, 4)
    game.run(game)
    return 0


main()
