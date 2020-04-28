import numpy as np

def test_initialize_game_state():
    from connect.common import initialize_game_state

    ret= initialize_game_state()

    assert  isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6,7)
    assert np.all(ret == 0)

def test_pretty_print_board(board:np.array):
    from connect.common import pretty_print_board

    ret = pretty_print_board()

    assert isinstance(ret, str)

def test_apply_player_action(
    board:np.array, action:PlayerAction, player: BoardPiece, copy: bool = False
):
    from connect.common import apply_player_action
    ret = apply_player_action()
    assert  isinstance(ret, np.ndarray)