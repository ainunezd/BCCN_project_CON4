import numpy as np
from random import randint
from agents.common import PlayerAction, BoardPiece, SavedState, PLAYER1, PLAYER2
from typing import Tuple, Optional


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
        #Tuple[PlayerAction, Optional[SavedState]]:

    action = randint(0,board.shape[1]-1)


    return np.int8(action), saved_state