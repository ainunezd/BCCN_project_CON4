import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, PLAYER1, PLAYER2
from typing import Tuple, Optional
from agents.common import apply_player_action, scoring_function

""" Weights order: 
  weight_middle 
  weight_2 
  weight_3 
  weight_win 

  weight_opp_2 
  weight_opp_3 
"""


def generate_move_scored(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
                         ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Return the best action for the player according to the score
    :param board: The board (np.ndarray)
    :param player: Player's turn (Boardpiece)
    :param saved_state: State of the game

    return best_action: Best action according to rewards
    return saved_state: State of the game
    """
    columns = board.shape[1]
    best_score = -10000
    best_action = np.int8

    weights_array = np.array([4, 2, 5, 1000, -2, -100])

    if board[0, columns // 2] == 0:
        best_action = columns // 2

    else:
        for c in range(columns):
            im_board, _ = apply_player_action(board=board, action=c, player=player, copy=True)
            score = scoring_function(board=im_board, weights=weights_array, player=player)
            if score > best_score:
                best_score = score
                best_action = c

    return best_action, saved_state
