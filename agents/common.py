##standard functions for multiple files

import numpy as np
from enum import Enum

from typing import Optional

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros((6, 7), dtype=BoardPiece)
    #board = initialize_game_state()
    #board[0, 0] = 2 #Lower left corner of board

def num_to_char(element):
    '''
    Helpful function for pretty print
    :param element: sting of the variables in the board to convert to nice figures like X or O instead of numbers
    :return: character for pretty print

    For zero--ascii 48
    For 1 ascii 49
    for 2 ascii 50

    for space ascii 32
    for X ascii 88
    for O ascii 79

    '''
    element_ascii = ord(element)
    if element_ascii == 48:
        element_ascii -= 16
    elif element_ascii == 49:
        element_ascii += 39
    elif element_ascii == 50:
        element_ascii += 29
    else:
        element_ascii = element_ascii
    character = chr(element_ascii)

    return character

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    a = board.astype(int)
    a = board.astype(str)

    pp_board = np.array([['/', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '/', '\n'],
                         ['/', num_to_char(a[5, 0]), ' ', num_to_char(a[5, 1]), ' ', num_to_char(a[5, 2]), ' ',num_to_char(a[5, 3]), ' ', num_to_char(a[5, 4]), ' ', num_to_char(a[5, 5]), ' ',num_to_char(a[5, 6]), ' ', '/', '\n'],
                         ['/', num_to_char(a[4, 0]), ' ', num_to_char(a[4, 1]), ' ', num_to_char(a[4, 2]), ' ',num_to_char(a[4, 3]), ' ', num_to_char(a[4, 4]), ' ', num_to_char(a[4, 5]), ' ',num_to_char(a[4, 6]), ' ', '/', '\n'],
                         ['/', num_to_char(a[3, 0]), ' ', num_to_char(a[3, 1]), ' ', num_to_char(a[3, 2]), ' ',num_to_char(a[3, 3]), ' ', num_to_char(a[3, 4]), ' ', num_to_char(a[3, 5]), ' ',num_to_char(a[3, 6]), ' ', '/', '\n'],
                         ['/', num_to_char(a[2, 0]), ' ', num_to_char(a[2, 1]), ' ', num_to_char(a[2, 2]), ' ',num_to_char(a[2, 3]), ' ', num_to_char(a[2, 4]), ' ', num_to_char(a[2, 5]), ' ',num_to_char(a[2, 6]), ' ', '/', '\n'],
                         ['/', num_to_char(a[1, 0]), ' ', num_to_char(a[1, 1]), ' ', num_to_char(a[1, 2]), ' ',num_to_char(a[1, 3]), ' ', num_to_char(a[1, 4]), ' ', num_to_char(a[1, 5]), ' ',num_to_char(a[1, 6]), ' ', '/', '\n'],
                         ['/', num_to_char(a[0, 0]), ' ', num_to_char(a[0, 1]), ' ', num_to_char(a[0, 2]), ' ',num_to_char(a[0, 3]), ' ', num_to_char(a[0, 4]), ' ', num_to_char(a[0, 5]), ' ',num_to_char(a[0, 6]), ' ', '/', '\n'],
                         ['/', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '=', '/', '\n'],
                         ['/', '0', ' ', '1', ' ', '2', ' ', '3', ' ', '4', ' ', '5', ' ', '6', ' ', '/', '\n']])
    pp_board = pp_board.flatten()
    pp_board = ''.join(pp_board)

    return pp_board



def char_to_num(element_ch):
    '''
    Helpful function for string to board
    :param element: sting of the variables in pretty board, convert to numbers
    :return: numbers on array board
    '''
    if element_ch == ' ':
        num = 0
    elif element_ch == 'X':
        num = 1
    elif element_ch == 'O':
        num = 2
    else:
        num = 5

    return num

def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """

    pp = np.array(list(pp_board)).reshape(9,17)
    board_ = np.array([[char_to_num(pp[6, 1]), char_to_num(pp[6, 3]), char_to_num(pp[6, 5]), char_to_num(pp[6, 7]), char_to_num(pp[6, 9]), char_to_num(pp[6, 11]), char_to_num(pp[6, 13])],
                      [char_to_num(pp[5, 1]), char_to_num(pp[5, 3]), char_to_num(pp[5, 5]), char_to_num(pp[5, 7]), char_to_num(pp[5, 9]), char_to_num(pp[5, 11]), char_to_num(pp[5, 13])],
                      [char_to_num(pp[4, 1]), char_to_num(pp[4, 3]), char_to_num(pp[4, 5]), char_to_num(pp[4, 7]), char_to_num(pp[4, 9]), char_to_num(pp[4, 11]), char_to_num(pp[4, 13])],
                      [char_to_num(pp[3, 1]), char_to_num(pp[3, 3]), char_to_num(pp[3, 5]), char_to_num(pp[3, 7]), char_to_num(pp[3, 9]), char_to_num(pp[3, 11]), char_to_num(pp[3, 13])],
                      [char_to_num(pp[2, 1]), char_to_num(pp[2, 3]), char_to_num(pp[2, 5]), char_to_num(pp[2, 7]), char_to_num(pp[2, 9]), char_to_num(pp[2, 11]), char_to_num(pp[2, 13])],
                      [char_to_num(pp[1, 1]), char_to_num(pp[1, 3]), char_to_num(pp[1, 5]), char_to_num(pp[1, 7]), char_to_num(pp[1, 9]), char_to_num(pp[1, 11]), char_to_num(pp[1, 13])]])
    return board_


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """

    num_rows = 6
    row = np.int8

    if copy:
        old_board = board.copy()
        return old_board
    else:
        pass

    column = int(action)

    if int(board[5, column]) == 0:
        for r in np.arange(num_rows-1, -1, -1):
            if board[r, column] == 0:
                row = int(r)

            else:
                pass
        board[row, column] = player
    else:
        pass

    new_board = board
    return new_board


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, ) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    columns = int(len(board[1]))
    rows = int(len(board))

    win = False

    # Check horizontal
    for c in range(columns - 3):
        for r in range(rows):
            if board[r, c] == player and board[r, c + 1] == player and board[r, c + 2] == player and board[
                r, c + 3] == player:
                win = True

    # check vertical
    for r in range(rows - 3):
        if board[r, last_action] == player and board[r + 1, last_action] == player and board[
            r + 2, last_action] == player and board[r + 3, last_action] == player:
            win = True

    # Check diagonal positive slope
    for c in range(columns - 3):
        for r in range(3, rows):
            if board[r, c] == player and board[r - 1, c + 1] == player and board[r - 2, c + 2] == player and board[
                r - 3, c + 3] == player:
                win = True

    # Check diagonal negative slope
    for c in range(columns - 3):
        for r in range(rows - 3):
            if board[r, c] == player and board[r + 1, c + 1] == player and board[r + 2, c + 2] == player and board[
                r + 3, c + 3] == player:
                win = True

    return win

def check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    elif board.all() !=0:
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING




from typing import Callable, Tuple

class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]