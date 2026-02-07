"""Python tests for the game theory module."""

import bellmaneq


class TestTicTacToe:
    """Tests for Tic-Tac-Toe."""

    def test_game_value_is_draw(self):
        game = bellmaneq.TicTacToe()
        board = [0] * 9
        value = game.minimax(board, 1, 9)
        assert abs(value) < 1e-10, f"Expected draw (0.0), got {value}"

    def test_best_move_exists(self):
        game = bellmaneq.TicTacToe()
        board = [0] * 9
        move = game.best_move(board, 1, 9)
        assert move is not None
        assert 0 <= move < 9

    def test_legal_actions_empty_board(self):
        game = bellmaneq.TicTacToe()
        board = [0] * 9
        actions = game.legal_actions(board, 1)
        assert len(actions) == 9

    def test_legal_actions_partial_board(self):
        game = bellmaneq.TicTacToe()
        board = [1, 2, 0, 0, 1, 0, 0, 0, 2]
        actions = game.legal_actions(board, 1)
        assert set(actions) == {2, 3, 5, 6, 7}

    def test_winning_move(self):
        """Should select a winning move when one is available."""
        game = bellmaneq.TicTacToe()
        # X has two in a row at cells (0, 1) -- cell 2 completes the win
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        move = game.best_move(board, 1, 9)
        assert move == 2, f"Expected winning move 2, got {move}"


class TestConnectFour:
    """Tests for Connect Four."""

    def test_empty_board(self):
        board = bellmaneq.ConnectFour.empty_board()
        assert len(board) == 6
        assert len(board[0]) == 7
        assert all(cell == 0 for row in board for cell in row)

    def test_legal_actions(self):
        game = bellmaneq.ConnectFour()
        board = bellmaneq.ConnectFour.empty_board()
        actions = game.legal_actions(board, 1)
        assert len(actions) == 7

    def test_apply_move(self):
        game = bellmaneq.ConnectFour()
        board = bellmaneq.ConnectFour.empty_board()
        new_board = game.apply_move(board, 1, 3)  # Drop piece in column 3
        # The piece should land in the bottom row (row 5), column 3
        assert new_board[5][3] == 1

    def test_check_winner_empty(self):
        game = bellmaneq.ConnectFour()
        board = bellmaneq.ConnectFour.empty_board()
        assert game.check_winner(board) == 0

    def test_blocks_threat(self):
        """Must not ignore a three-in-a-row threat."""
        game = bellmaneq.ConnectFour()
        board = bellmaneq.ConnectFour.empty_board()
        # Opponent has three pieces stacked in column 0
        board[5][0] = 2
        board[4][0] = 2
        board[3][0] = 2
        best = game.best_move(board, 1, 4)
        assert best == 0, f"Should block at column 0, got {best}"

    def test_best_move_shallow(self):
        game = bellmaneq.ConnectFour()
        board = bellmaneq.ConnectFour.empty_board()
        move = game.best_move(board, 1, 4)
        assert move is not None
        assert 0 <= move < 7
