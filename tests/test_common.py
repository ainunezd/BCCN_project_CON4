import numpy as np
import pytest

from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2

# Boards for test
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
                    [0, 0, 2, 2, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 2, 0, 0, 0]], dtype=BoardPiece)

board_4 = np.array([[2, 1, 2, 1, 2, 2, 2, 1],
                    [1, 1, 1, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 2, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=BoardPiece)

board_to_test = board_3
play_act = np.int8(2)
play = PLAYER2


def test_initialize_game_state():
    """
    Test the correct initialization in zeros of the board
    """
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board(board=board_to_test):
    """
    Test the printing of the board and also the function of num:to char included
    :param board : Selected board
    """
    from agents.common import pretty_print_board
    ret = pretty_print_board(board)
    assert isinstance(ret, str)
    print(ret)


def test_string_to_board(board=board_to_test):
    """
    Test the implementation of the function string to board
    :param board: Selected board
    """
    from agents.common import string_to_board, pretty_print_board

    ret_test = string_to_board(pretty_print_board(board))
    assert isinstance(ret_test, np.ndarray)
    assert ret_test.dtype == BoardPiece
    comparison = board == ret_test
    comparison.all()
    assert comparison.all()


def test_valid_action_yes(board=board_to_test, action=play_act):
    """
    Test the function valid action for true valid actions
    :param board:
    :param action: Selected action between the number of columns in the board
    """
    from agents.common import valid_action
    assert valid_action(board, action)


def test_valid_action_no(board=board_3, action=3):
    """
    Test the function valid action for false valid actions
    :param board: Board selected
    :param action: Selected action between the number of colums in the board
    """
    from agents.common import valid_action
    assert not valid_action(board, action)


def test_apply_player_action(board=board_to_test, action=play_act, player=play, copy=False):
    """
    Function to test the applyed action in the bpard by a player
    :param board: Playing board (np.ndarray)
    :param action: The column where the piece is placed (PlayerAction)
    :param player: The player putting the piece (BoardPiece)
    :param copy: If we want to make a copy of the board, Optional (bool)
    """
    from agents.common import apply_player_action, valid_action

    c_ret, ret = apply_player_action(board, action, player)
    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    if valid_action(c_ret, action):
        free = np.argwhere(c_ret[:, action] == NO_PLAYER)
        check = c_ret[int(free[0])-1, action] == player
    else:
        check = c_ret[-1, action] == player
    assert check


def test_apply_player_action_exc(board=board_to_test, action=play_act, player=play, copy=False):
    """
    Function to test the applyed action in the board by a player
    :param board: Playing board (np.ndarray)
    :param action: The column where the piece is placed (PlayerAction)
    :param player: The player putting the piece (BoardPiece)
    :param copy: If we want to make a copy of the board, Optional (bool)
    """
    from agents.common import apply_player_action

    ret = np.ones((board.shape[0], board.shape[1]))
    with pytest.raises(Exception, match='Piece can not be placed in this column'):
        apply_player_action(ret, action, player)


def test_apply_player_action_copy(board=board_to_test, action=play_act, player=play, copy=True):
    """
    Function to test the applyed action in the board by a player
    :param board: Playing board (np.ndarray)
    :param action: The column where the piece is placed (PlayerAction)
    :param player: The player putting the piece (BoardPiece)
    :param copy: If we want to make a copy of the board, Optional (bool)
    """
    from agents.common import apply_player_action

    c_ret, ret = apply_player_action(board, action, player, copy)
    assert isinstance(ret, np.ndarray)
    assert isinstance(c_ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert c_ret.dtype == BoardPiece
    assert np.any(ret != NO_PLAYER)
    comparison = board == ret
    comparison.all()
    assert comparison.all()
    comparison2 = board == c_ret
    comparison2.all()
    assert not comparison2.all()


def test_connected_four(board=board_to_test, player=play, last_action=play_act):
    """
    Fuction to determine if the last piece placed is connecting 4 of the same
    :param board: Playing board (np.ndarray)
    :param player: The player putting the piece (BoardPiece)
    :param last_action: The column where the piece is placed (PlayerAction)
    """
    from agents.common import connected_four

    ret = connected_four(board, player, last_action)

    assert isinstance(ret, bool)


def test_full_board_yes(player=play):
    """
    Test to check if the board is full
    :param player: The player putting the piece (BoardPiece)
    """
    from agents.common import initialize_game_state, full_board

    ret = initialize_game_state()
    ret[-1, :] = player
    assert full_board(ret)


def test_full_board_no(board=board_to_test, player=play):
    """
    Test to check if the board is NOT full
    :param board: Selected board (np.ndarray)
    """
    from agents.common import initialize_game_state, full_board

    ret = initialize_game_state()
    ret[-1, 1:-1] = player
    assert not full_board(ret)
    assert not full_board(board)


def test_check_end_state_draw(board=board_to_test, player=play, last_action=play_act):
    """
    Function to determine DRAW game state
    :param board: Playing board (np.ndarray)
    :param player: The player putting the piece (BoardPiece)
    :param last_action: The column where the piece is placed (PlayerAction)
    """
    from agents.common import check_end_state, GameState

    ret = 3 * np.ones((board.shape[0], board.shape[1]))
    assert check_end_state(ret, player, last_action) == GameState.IS_DRAW


def test_check_end_state_win_horizontal():
    """
    Function to determine WIN horizontal game state
    """
    from agents.common import initialize_game_state, check_end_state, GameState

    ret = initialize_game_state()
    ret[0, 1:5] = PLAYER2
    assert check_end_state(ret, PLAYER2, 2) == GameState.IS_WIN
    assert not check_end_state(ret, PLAYER1, 2) == GameState.IS_WIN


def test_check_end_state_win_vertical():
    """
    Fuction to determine WIN vertical game state
    :param player: The player putting the piece (BoardPiece)
    """
    from agents.common import initialize_game_state, check_end_state, GameState

    ret = initialize_game_state()
    ret[0:4, 2] = PLAYER2
    assert check_end_state(ret, PLAYER2, 2) == GameState.IS_WIN
    assert not check_end_state(ret, PLAYER1, 2) == GameState.IS_WIN


def test_check_end_state_win_diagonal_1():
    """
    Fuction to determine WIN negative diagonal game state
    :param player: The player putting the piece (BoardPiece)
    """
    from agents.common import initialize_game_state, check_end_state, GameState

    ret = initialize_game_state()
    ret[0, 0] = PLAYER2
    ret[1, 1] = PLAYER2
    ret[2, 2] = PLAYER2
    ret[3, 3] = PLAYER2
    assert check_end_state(ret, PLAYER2, 2) == GameState.IS_WIN


def test_check_end_state_win_diagonal_2():
    """
    Fuction to determine WIN positive diagonal game state
    :param player: The player putting the piece (BoardPiece)
    """
    from agents.common import initialize_game_state, check_end_state, GameState

    ret = initialize_game_state()
    ret[4, 0] = PLAYER2
    ret[3, 1] = PLAYER2
    ret[2, 2] = PLAYER2
    ret[1, 3] = PLAYER2
    assert check_end_state(ret, PLAYER2, 2) == GameState.IS_WIN


def test_eval_window(player = play):
    """
    Test of the score evaluation for one window
    weights_array = np.array([4, 2, 5, 1000, -2, -100])
        weight_middle, weight_2, weight_3, weight_win, weight_opp_2, weight_opp_3
    :param window:
    :param weights:
    :return:
    """
    from agents.common import eval_window

    weights_array = np.array([4, 2, 5, 1000, -2, -100])
    win = np.array([0, 2, 2, 0])  # 2 For both player2 and -2 for player1
    if player == PLAYER1:
        assert eval_window(win, weights_array, player) == -2
    else: assert eval_window(win, weights_array, player) == 2

    win1 = np.array([1, 2, 2, 1]) # 0 For both players
    assert eval_window(win1, weights_array, player) == 0

    win2 = np.array([0, 2, 2, 2])
    if player == PLAYER1:
        assert eval_window(win2, weights_array, player) == -100
    else: assert eval_window(win2, weights_array, player) == 5

    win3 = np.array([2, 2, 2, 2])
    if player == PLAYER2:
        assert eval_window(win3, weights_array, player) == 1000


def test_scoring_function():
    """
    Test of the score evaluation of the hole board
    weights_array = np.array([4, 2, 5, 1000, -2, -100])
        weight_middle, weight_2, weight_3, weight_win, weight_opp_2, weight_opp_3
    """
    from agents.common import initialize_game_state, scoring_function

    # weight_2 and opp_2
    weights_array = np.array([4, 2, 5, 1000, -2, -100])
    ret = initialize_game_state()
    ret[0,2]=PLAYER2
    ret[0,1]=PLAYER2
    assert scoring_function(ret, weights_array, PLAYER2)==4
    assert scoring_function(ret, weights_array, PLAYER1)==-4

    # weight_3 and opp_3
    ret2 = initialize_game_state()
    ret2[0, 1] = PLAYER2
    ret2[1, 1] = PLAYER2
    ret2[2, 1] = PLAYER2
    assert scoring_function(ret2, weights_array, PLAYER2) == 7
    assert scoring_function(ret2, weights_array, PLAYER1) == -102

    # weight_win
    ret3 = initialize_game_state()
    ret3[0, 0] = PLAYER2
    ret3[1, 1] = PLAYER2
    ret3[2, 2] = PLAYER2
    ret3[3, 3] = PLAYER2
    assert scoring_function(ret3, weights_array, PLAYER2) == 1007


def test_available_columns(player = play):
    """
    Test the function to obtain which columns are available to play as array
    """
    from agents.common import initialize_game_state, available_columns
    ret = initialize_game_state()
    ret[-1, 2:4] = player
    av_c = available_columns(ret)
    assert isinstance(av_c, np.ndarray)
    assert (av_c == np.array([0,1,4,5,6])).all()

    ret[-1, 6] = player
    assert (available_columns(ret) == np.array([0,1,4,5])).all()


