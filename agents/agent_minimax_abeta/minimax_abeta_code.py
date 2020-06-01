import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, PLAYER1, PLAYER2, weights_array, NO_PLAYER
from typing import Tuple, Optional
from agents.common import connected_four, apply_player_action, scoring_function
from random import randint


num_depth = 4


def minimax_ab(board: np.ndarray, depth: int, maximizingPlayer: bool, player:BoardPiece, weights: np.ndarray = weights_array):
    '''Minimax function with alpha-beta prunning optimization
    to obtain the agent best move'''
    board_terminal = connected_four(board, player=PLAYER1) or connected_four(board, player=PLAYER2)

    columns = np.argwhere(board[-1,:] == NO_PLAYER)
    alpha = -1000000000
    beta = 1000000000
    if depth == 0 or board_terminal:
        board_score = scoring_function(board=board, weights=weights, player=player)
        return board_score, 0


    elif maximizingPlayer:
        board_score = -10000000
        for c in columns:
            im_board = apply_player_action(board=board, action=c, player=player, copy=True)
            if im_board[-1, c] != 0:
                board_terminal = True
            score, _ = minimax_ab(im_board, depth - 1, False, player, weights)


            if score > board_score:
                board_score = score
                best_move = c
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return board_score, best_move

    else:
        board_score = 100000000
        if player == PLAYER1:
            opponent = PLAYER2
        else:
            opponent = PLAYER1
        for c in columns:
            im_board = apply_player_action(board=board, action=c, player=opponent, copy=True)
            if im_board[-1, c] != 0:
                board_terminal = True
            score, _ = minimax_ab(im_board, depth - 1, True, opponent, weights)
            score= -score
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


    if board[0,columns // 2] == 0:
        best_column = columns // 2
    else:
        score, best_column = minimax_ab(board = board, depth = num_depth, maximizingPlayer = True, player=player)


    return best_column, saved_state


