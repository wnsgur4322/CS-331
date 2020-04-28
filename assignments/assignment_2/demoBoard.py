#This file exists to display some board functionality
#Erich Kramer 4/26/18
#
#
#
# CS 331 - Spring 2020
# Programming Assignment 2 - Simplified Othello
# Junhyeok Jeong, jeongju@oregonstate.edu


from Board import Board


x = Board(15, 15)


x.set_cell( 4, 4, 'x')

x.set_cell( 1, 3, 'B')


x.display()

y = x.cloneBoard()
y.display()
