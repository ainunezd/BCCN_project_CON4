import numpy as np
import pytest

from agents.common import PLAYER1, PLAYER2, initialize_game_state
from agents.agent_minimax_aa.agent_minimax_code import generate_move_minimax, minimax


def test_generate_move_minimax():
    """
    Test for the first movement of the minimax agent
    """
    ret = initialize_game_state()
    column, state = generate_move_minimax(ret, PLAYER1)
    assert int(column) == 3

    ret2 = initialize_game_state()
    column, state = generate_move_minimax(ret2, PLAYER2)
    assert int(column) == 3


def test_minimax_draw():
    """
    Function to test minimax scoring when is draw and the board is full
    """
    ret = 3 * np.ones((6, 7))
    score, column = minimax(ret, 0, True, PLAYER1)
    assert score == 0
    assert column is None

    ret2 = 3 * np.ones((6, 7))
    score, column = minimax(ret2, 0, True, PLAYER2)
    assert score == 0
    assert column is None


def test_minimax_win_lose():
    """
    Test the minimax when the agent loses or wins
    """
    ret = initialize_game_state()
    ret[0, 1:5] = PLAYER1
    print(ret)
    score, column = minimax(ret, 0, True, PLAYER1)
    assert score == 1012  # 5+ 1000+ 5 + 2
    assert column is None

    score, column = minimax(ret, 0, True, PLAYER2)
    print(score)
    assert score == -202  # -100 -100 -2
    assert column is None
