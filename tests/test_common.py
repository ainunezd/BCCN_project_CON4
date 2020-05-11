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

board_to_test = board_2
PlayerAction = 3



def test_pretty_print_board(board=board_to_test):
    from agents.common import pretty_print_board

    ret = pretty_print_board(board)

    assert isinstance(ret, str)
    print(ret)


'''def test_string_to_board(board = pretty_print_board(board_3)):
    from agents.common import string_to_board

    ret = string_to_board(board)

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)

    print(ret)'''


def test_apply_player_action(board=board_to_test, action=PlayerAction, player=PLAYER1, copy=False):
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