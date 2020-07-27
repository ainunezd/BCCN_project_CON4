import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, PLAYER1, PLAYER2, weights_array, NO_PLAYER
from typing import Tuple, Optional
from agents.common import connected_four, apply_player_action, scoring_function, full_board
from random import randint


num_depth = 4


def minimax(board: np.ndarray, depth: int, maximizingPlayer: bool, player:BoardPiece, weights: np.ndarray = weights_array):
    """
    Minimax function to obtain the best position for a given player
    :param board: Actual board in the game (np.ndarray)
    :param depth: Depth for simulation
    :param maximizingPlayer: Flag (bool) for maximizing or minimizing
    :param player: Player that is being maximize value or not
    :param weights: Array of weights for the scoring function
    :return: score of the board and the best move to perform
    """

    board_terminal = connected_four(board, player=PLAYER1) or connected_four(board, player=PLAYER2) or full_board(board)
    columns = np.argwhere(board[-1, :] == NO_PLAYER)

    if depth == 0 or board_terminal:
        board_score = scoring_function(board=board, weights=weights, player=player)
        return int(board_score), None

    elif maximizingPlayer:
        board_score = -10000000
        for c in columns:
            c = int(c)
            im_board, _ = apply_player_action(board=board, action=c, player=player, copy=True)
            if im_board[-1, c] != 0:
                board_terminal = True
            score, _ = minimax(im_board, depth - 1, False, player, weights)
            if score > board_score:
                board_score = score
                best_move = c
        return int(board_score), int(best_move)

    else:
        board_score = 100000000
        if player == PLAYER1:
            opponent = PLAYER2
        else:
            opponent = PLAYER1
        for c in columns:
            c = int(c)
            im_board, _ = apply_player_action(board=board, action=c, player=opponent, copy=True)
            if im_board[-1, c] != 0:
                board_terminal = True
            score, _ = minimax(im_board, depth - 1, True, opponent, weights)
            score = -score
            if score < board_score:
                board_score = score
                best_move = c
        return int(board_score), int(best_move)


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Tuple[PlayerAction, Optional[SavedState]]:

    columns = board.shape[1]

    if board[0, columns // 2] == 0:
        best_column = columns // 2
    else:
        score, best_column = minimax(board=board, depth=num_depth, maximizingPlayer=True, player=player)

    return int(best_column), saved_state
