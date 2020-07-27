import numpy as np
import pytest

from agents.common import PLAYER1, PLAYER2, initialize_game_state
from agents.agent_minimax_ab.minimax_ab import minimax_ab, generate_move_minimax_alphabeta

def test_generate_move_minimax_alphabeta():
    '''
    Test for the first movement of the minimax alpha-beta agent
    '''
    ret = initialize_game_state()
    column, state = generate_move_minimax_alphabeta(ret, PLAYER1,[])
    assert int(column) == 3

    ret2 = initialize_game_state()
    column, state = generate_move_minimax_alphabeta(ret2, PLAYER2, [])
    assert int(column) == 3

def test_minimax_ab_draw():
    '''
    Function to test minimax alpha-beta scoring when is draw and the board is full
    '''
    ret = 3 * np.ones((6, 7))
    score, column = minimax_ab(ret, 0, True, PLAYER1)
    assert score == 0
    assert column == None

    ret2 = 3 * np.ones((6, 7))
    score, column = minimax_ab(ret2, 0, True, PLAYER2)
    assert score == 0
    assert column == None

def test_minimax_ab_win_lose():
    '''
    Test the minimax alpha-beta when the agent loses or wins
    '''
    ret = initialize_game_state()
    ret[0, 1:5] = PLAYER1
    score, column = minimax_ab(ret, 0, True, PLAYER1)
    assert score == 1012 # 5+ 1000+ 5 + 2
    assert column == None

    score, column = minimax_ab(ret, 0, True, PLAYER2)
    assert score == -202  # -100 -100 -2
    assert column == None