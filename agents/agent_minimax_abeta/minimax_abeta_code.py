import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, PLAYER1, PLAYER2, weights_array
from typing import Tuple, Optional
from agents.common import connected_four, apply_player_action, scoring_function
from random import randint


num_depth = 3


def minimax_ab(board: np.ndarray, depth: int, maximizingPlayer: bool, weights: np.ndarray = weights_array):
    board_terminal = connected_four(board, player=PLAYER1) or connected_four(board, player=PLAYER2)
    columns = board.shape[1]
    best_move = np.random.randint(0, columns - 1)
    alpha = -1000000000
    beta = 1000000000
    if depth == 0 or board_terminal:
        board_score = scoring_function(board=board, weights=weights)
        return board_score, best_move

    elif maximizingPlayer:
        board_score = -10000000
        for c in range(columns):
            im_board = apply_player_action(board=board, action=c, player=PLAYER1, copy=True)
            if im_board[-1, c] != 0:
                board_terminal = True
            score, _ = minimax_ab(im_board, depth - 1, False, weights)

            if score > board_score:
                board_score = score
                best_move = c
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return board_score, best_move

    else:
        board_score = 100000000
        for c in range(columns):
            im_board = apply_player_action(board=board, action=c, player=PLAYER2, copy=True)
            if im_board[-1, c] != 0:
                board_terminal = True
            score, _ = minimax_ab(im_board, depth - 1, True, weights)
            if score < board_score:
                board_score = score
                best_move = c
            beta = min(beta, score)
            if alpha >= beta:
                break
        return board_score, best_move



def generate_move_minimax_alphabeta(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
        #Tuple[PlayerAction, Optional[SavedState]]:
    best_column = randint(0,6)
    columns = board.shape[1]

    if player ==PLAYER1:
        if board[0,columns // 2] == 0:
            best_column = columns // 2
        else:
            score, best_column = minimax_ab(board = board, depth = num_depth, maximizingPlayer = True)


    return best_column, saved_state


