import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2

def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


### Boards for test
board_1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=BoardPiece)

board_2 = np.array([[1, 2, 1, 2, 1, 1, 2],
                    [2, 1, 2, 0, 1, 1, 0],
                    [1, 2, 0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=BoardPiece)

board_3 = np.array([[2, 1, 2, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2, 2],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=BoardPiece)

board_4 = np.array([[2, 1, 2, 1, 2, 2, 2, 1],
                    [1, 1, 1, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=BoardPiece)

board_to_test = board_3
PlayerAction = 2



def test_pretty_print_board(board=board_to_test):
    from agents.common import pretty_print_board

    ret = pretty_print_board(board)

    assert isinstance(ret, str)
    print(ret)


def test_string_to_board(board = board_to_test):
    from agents.common import string_to_board, pretty_print_board

    ret = string_to_board(pretty_print_board(board))

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)

    print(ret)


def test_apply_player_action(board=board_to_test, action=PlayerAction, player=PLAYER2, copy=False):
    from agents.common import apply_player_action

    ret = apply_player_action(board, action , player , copy)
    print(ret)
    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.any(ret != NO_PLAYER)
    print(ret)

def test_connected_four(board = board_to_test, player = PLAYER2, last_action = PlayerAction):
    from agents.common import connected_four
    
    ret = connected_four(board, player, last_action)

    assert isinstance(ret, bool)
    print(ret)

def test_check_end_state(board = board_to_test, player =  PLAYER2, last_action= PlayerAction):
    from agents.common import check_end_state

    ret = check_end_state(board, player, last_action)
    print(ret)

win_1 = [0,0,1,1]
weights_array = np.array([4, 2, 5, 1000, -2, -100])

def test_eval_window(window = win_1, weights =weights_array):
    from agents.common import eval_window

    ret = eval_window(window, weights)

    assert isinstance(ret, int)
    print(ret)


def test_scoring_function ( board = board_to_test, weights =weights_array):
    from agents.common import scoring_function

    ret = scoring_function(board, weights)

    assert isinstance(ret, int)
    print(ret)