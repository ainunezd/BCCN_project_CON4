import numpy as np
import pytest

from agents.common import PLAYER1, PLAYER2

def test_generate_move_scored():
    '''
    Test for the first movement of the scored agent
    '''
    from agents.agent_score.score_aa import generate_move_scored
    from agents.common import initialize_game_state
    ret = initialize_game_state()
    column, state = generate_move_scored(ret, PLAYER1,[])
    assert int(column) == 3

    ret2 = initialize_game_state()
    column, state = generate_move_scored(ret2, PLAYER2, [])
    assert int(column) == 3
