# standard functions for multiple files
import numpy as np
from enum import Enum
from typing import Optional
from typing import Callable, Tuple

weights_array = np.array([4, 2, 5, 1000, -2, -100])

BoardPiece = np.int8  # The data type (d type) of the board
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
    Returns an nd array, shape (6, 7) and data type (d type) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros((6, 7), dtype=BoardPiece)
    # board = initialize_game_state()
    # board[0, 0] = 2 #Lower left corner of board


def num_to_char(element) -> str:
    """
    Helpful function for pretty print
    :param element: sting of the variables in the board to convert to nice figures like X or O instead of numbers
    :return: character for pretty print
    For zero--ascii 48
    For 1 ascii 49
    for 2 ascii 50
    for space ascii 32
    for X ascii 88
    for O ascii 79
    """
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
    rows_board = board.shape[0]
    cols_board = board.shape[1]
    pp_board = np.zeros((rows_board + 3, cols_board * 2 + 3))

    board = board.astype(int)
    board = board.astype(str)
    pp_board = pp_board.astype(int)
    pp_board = pp_board.astype(str)

    pp_board[:, 0] = '/'
    pp_board[:, -1] = '\n'
    pp_board[:, -2] = '/'
    pp_board[0, 1:-2] = '='
    pp_board[-2, 1:-2] = '='

    i = 0

    for r in np.arange(1, pp_board.shape[0], 1):
        for c in np.arange(1, pp_board.shape[1] - 2, 1):
            if c % 2 == 0:
                pp_board[r, c] = ' '
            else:
                if r == (pp_board.shape[0] - 1):
                    pp_board[r, c] = str(i)
                    i += 1
                elif 1 <= r <= rows_board:
                    pp_board[r, c] = num_to_char(board[abs(r - rows_board), (c - 1) // 2])

    pp_board = pp_board.flatten()
    pp_board = ''.join(pp_board)

    return pp_board


def char_to_num(element_ch) -> int:
    """
    Helpful function for string to board
    :param element_ch: sting of the variables in pretty board, convert to numbers
    :return: numbers on array board
    """
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

    pp_rows = len(pp_board.splitlines())
    pp_cols = len(pp_board) // pp_rows

    board_rows = pp_rows - 3
    board_cols = (pp_cols - 3) // 2

    pp = np.array(list(pp_board)).reshape(pp_rows, pp_cols)
    board_ = np.zeros((board_rows, board_cols), dtype=BoardPiece)

    for r in range(board_rows):
        for c in range(board_cols):
            board_[r, c] = char_to_num(pp[abs(r - board_rows), c * 2 + 1])

    return board_.astype('int8')


def valid_action(board: np.ndarray, action: PlayerAction) -> bool:
    """
    Helpful function to determine if there is still space in the column for an action
    :param board: The board (np.nd array)
    :param action: The place where the next move of a player is place (BoardPiece)
    :return : Boolean, true for valid action, false otherwise
    """
    return board[-1, action] == NO_PLAYER


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> tuple:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    :param board: Playing board (np.nd array)
    :param action: The column where the piece is placed (PlayerAction)
    :param player: The player putting the piece (BoardPiece)
    :param copy: If we want to make a copy of the board, Optional (bool)
    :return: Copied board and original board (np.nd array)
    """
    if valid_action(board, action):
        if copy:
            c_board = np.copy(board)
        else:
            c_board = board
        free = np.argwhere(c_board[:, action] == NO_PLAYER)
        c_board[int(free[0]), action] = player
    else:
        raise Exception('Piece can not be placed in this column')

    return c_board, board


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, ) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    :param board: Playing board (np.nd array)
    :param player: The player putting the piece (BoardPiece)
    :param last_action: The column where the piece is placed (PlayerAction)
    :return win: Flag of true for the win state
    """
    columns = board.shape[1]
    rows = board.shape[0]

    win = False

    # Check horizontal
    for c in range(columns - 3):
        for r in range(rows):
            if all(board[r, c:c + 4] == player):
                win = True

    # check vertical
    for c in range(columns):
        for r in range(rows - 3):
            if all(board[r:r + 4, c] == player):
                win = True

    # Check diagonal positive slope
    for c in range(columns - 3):
        for r in range(3, rows):
            if board[r, c] == player and board[r - 1, c + 1] == player and board[r - 2, c + 2] == player and board[r - 3, c + 3] == player:
                win = True

    # Check diagonal negative slope
    for c in range(columns - 3):
        for r in range(rows - 3):
            if board[r, c] == player and board[r + 1, c + 1] == player and board[r + 2, c + 2] == player and board[r + 3, c + 3] == player:
                win = True

    return win


def full_board(board: np.ndarray) -> bool:
    """
    Function to check if the board is full
    :param board: Playing board (np.ndarray)
    :return: True if the board is full, False otherwise (bool)
    """
    if any(board[-1, :] == 0):
        return False
    else:
        return True


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, ) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    :param board: Playing board (np.ndarray)
    :param player: The player putting the piece (BoardPiece)
    :param last_action: The column where the piece is placed (PlayerAction)
    :return: End state of the game after a move
    """
    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    elif full_board(board):
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def eval_window(window: np.ndarray, weights: np.ndarray, player: BoardPiece) -> int:
    """
    Returns the score for a window of 4 positions in the board
    :param window: Also called patch, is an array of 4 inside the board
    :param weights: Weights for the scoring (np.ndarray)
    :param player: Actual player (BoardPiece)

    :return score: Score of this window (int)
    """

    score = 0
    if player == PLAYER1:
        opponent = PLAYER2
    else:
        opponent = PLAYER1

    if np.count_nonzero(window == player) == 4:
        score += weights[3]
    elif np.count_nonzero(window == player) == 3 and np.count_nonzero(window == NO_PLAYER) == 1:
        score += weights[2]
    elif np.count_nonzero(window == player) == 2 and np.count_nonzero(window == NO_PLAYER) == 2:
        score += weights[1]
    elif np.count_nonzero(window == opponent) == 2 and np.count_nonzero(window == NO_PLAYER) == 2:
        score += weights[4]
    elif np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == NO_PLAYER) == 1:
        score += weights[5]

    return int(score)


def scoring_function(board: np.ndarray, weights: np.ndarray, player: BoardPiece) -> int:
    """
    To calculate the score of the full board
    Weights order:
        weight_middle
        weight_2
        weight_3
        weight_win

        weight_opp_2
        weight_opp_3
    :param board: Playing board (np.ndarray)
    :param weights: Weights for the scoring (np.ndarray)
    :param player: Actual player

    :return total_score: Score of the full board
    """
    columns = board.shape[1]
    rows = board.shape[0]

    # Score horizontal
    score_horizontal = 0
    for r in range(rows):
        for c in range(columns - 3):
            patch = board[r, c:c + 4]
            score_horizontal += eval_window(window=patch, weights=weights, player=player)

    # Score vertical
    score_vertical = 0
    for c in range(columns):
        for r in range(rows - 3):
            patch = board[r: r + 4, c]
            score_vertical += eval_window(window=patch, weights=weights, player=player)

    # Score diagonal positive
    score_diag_pos = 0
    for c in range(columns - 3):
        for r in range(rows - 3):
            patch = np.array([board[r, c], board[r + 1, c + 1], board[r + 2, c + 2], board[r + 3, c + 3]])
            score_diag_pos += eval_window(window=patch, weights=weights, player=player)

    # Score diagonal negative
    score_diag_neg = 0
    for c in range(columns - 3):
        for r in range(3, rows):
            patch = np.array([board[r, c], board[r - 1, c + 1], board[r - 2, c + 2], board[r - 3, c + 3]])
            score_diag_neg += eval_window(window=patch, weights=weights, player=player)

    total_score = score_horizontal + score_vertical + score_diag_neg + score_diag_pos
    return int(total_score)


def available_columns(board: np.ndarray) -> np.ndarray:
    """
    Function to obtain an array of the columns where a piece can be played
    :param board: Board played (np.ndarray)
    :return av_columns: Columns where a piece can be played (np.ndarray)
    """
    free = np.argwhere(board[-1, :] == NO_PLAYER)
    av_columns = free[:, 0].astype('int8')
    return av_columns
