##standard functions for multiple files

import numpy as np
from enum import Enum

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8



def initialize_game_state()-> np.array:
    return  np.zeros((6, 7), dtype=BoardPiece)
board = initialize_game_state()
board[0, 0] = 2 #Lower left corner of board

def pretty_print_board(board:np.array)->str:
    pass

def apply_player_action(
    board:np.array, action:PlayerAction, player: BoardPiece, copy: bool = False
) ->np.ndarray:
    pass

def string_to_board(pp_board: str)->np.ndarray:
    pass




#from typing import union
#a:union[int,str]

#def add(a,b):
 #   return a+b

#print(add('x','y'))
#print(add(1,2))


