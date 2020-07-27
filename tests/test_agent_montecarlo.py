import numpy as np
import pytest
from random import choice

from agents.common import PLAYER1, PLAYER2, NO_PLAYER, BoardPiece, initialize_game_state
from agents.agent_montecarlo.montecarlo import node_montecarlo as node_mc_, best_child, expansion
from agents.agent_montecarlo.montecarlo import default_policy

# Boards for test

board_1 = np.array([[1, 2, 1, 2, 1, 1, 2],
                    [2, 1, 0, 0, 1, 1, 0],
                    [1, 2, 0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=BoardPiece)

board_2 = np.array([[1, 2, 1, 2, 1, 1, 2],
                    [2, 1, 0, 0, 1, 1, 0],
                    [1, 2, 0, 0, 2, 0, 0],
                    [2, 0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 2, 0, 0],
                    [2, 0, 0, 0, 1, 0, 0]], dtype=BoardPiece)

board_3 = np.array([[1, 2, 1, 2, 2, 1, 2],
                    [2, 1, 2, 0, 1, 2, 2],
                    [1, 1, 2, 0, 2, 0, 1],
                    [2, 0, 2, 0, 1, 0, 2],
                    [1, 0, 1, 0, 2, 0, 1],
                    [2, 0, 2, 0, 1, 0, 2]], dtype=BoardPiece)

board_4 = np.array([[1, 1, 2, 2, 1, 2, 0],
                    [2, 2, 1, 1, 2, 1, 0],
                    [1, 1, 2, 2, 1, 2, 0],
                    [2, 2, 1, 1, 2, 1, 0],
                    [1, 1, 2, 2, 1, 2, 0],
                    [2, 2, 1, 1, 2, 1, 0]], dtype=BoardPiece)


def test_child_addition(board=board_2):
    """
    Test for checking the array of children in the node depending on how are the
    getting added after trying an action there
    :param board: Test board (np.ndarray)
    """
    test_node = node_mc_(board_state=board, player=PLAYER1)
    assert isinstance(test_node.available_actions, np.ndarray)
    assert (test_node.available_actions == np.array([1, 2, 3, 5, 6])).all()
    assert len(test_node.available_actions) == 5
    assert len(test_node.children) == 0
    action = 3
    test_child = test_node.child_addition(action=action)
    assert test_child == test_node.children[0]
    assert (test_node.available_actions == np.array([1, 2, 5, 6])).all()
    assert len(test_node.available_actions) == 4
    assert len(test_node.children) == 1
    assert isinstance(test_child, object)


def test_free_columns_still_playing():
    """
    Testing the correct return of available columns (actions) for the player to perform
    """
    test_node1 = node_mc_(board_state=board_1, player=PLAYER1)
    assert (test_node1.free_columns() == np.array([0, 1, 2, 3, 4, 5, 6])).all()
    test_node2 = node_mc_(board_state=board_2, player=PLAYER1)
    assert (test_node2.free_columns() == np.array([1, 2, 3, 5, 6])).all()


def test_free_columns_draw():
    """
    Testing that there should not be return for draw
    """
    board = 3 * np.ones((6, 7))
    test_node = node_mc_(board_state=board, player=PLAYER1)
    assert len(test_node.free_columns()) == 0
    assert isinstance(test_node.free_columns(), np.ndarray)


def test_free_columns_win():
    board = initialize_game_state()
    board[0:4] = PLAYER2
    test_node = node_mc_(board_state=board, player=PLAYER1)
    assert len(test_node.free_columns()) == 0
    assert isinstance(test_node.free_columns(), np.ndarray)


def test_best_child():
    """
    Test for getting the best child node from a node.
    Evaluation of the UCB per node.
    In this case the exploration term is 1
    """
    test_node = node_mc_(board_state=board_3, player=PLAYER1)
    assert (test_node.free_columns() == np.array([1, 3, 5])).all()
    child_board1 = board_3.copy()
    child_board2 = board_3.copy()
    child_board3 = board_3.copy()
    child_board1[3, 1] = PLAYER1
    child_board2[1, 3] = PLAYER1
    child_board3[2, 5] = PLAYER1
    child_node1 = node_mc_(board_state=child_board1, player=PLAYER2)
    child_node2 = node_mc_(board_state=child_board2, player=PLAYER2)
    child_node3 = node_mc_(board_state=child_board3, player=PLAYER2)
    test_node.visits = 100
    child_node1.visits = 10
    child_node2.visits = 20
    child_node3.visits = 50
    child_node1.reward = 8
    child_node2.reward = 18
    child_node3.reward = 30
    test_node.children = np.array([child_node1, child_node2, child_node3])
    assert (best_child(test_node, c_value = 1) == child_node1)


def test_expansion():
    """
    Function to check the creation of a child from random action selection
    """
    board = 3* np.ones((6,7))
    board[-1, 2] = NO_PLAYER
    test_node = node_mc_(board_state=board.copy(), player=PLAYER1)
    test_child = expansion(test_node)
    child_test_node = test_node.child_addition(2)
    assert (test_child.board_state == child_test_node.board_state).all()
    assert isinstance(test_child, object)


def test_default_policy():
    test_node = node_mc_(board_state=board_4, player=PLAYER1)
    assert default_policy(test_node) == 1
    test_node2 = node_mc_(board_state=board_4, player=PLAYER2)
    assert default_policy(test_node2) == 0
