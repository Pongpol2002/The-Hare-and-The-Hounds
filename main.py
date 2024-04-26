import sys
import numpy as np
from scipy.spatial.distance import cityblock
import math

board = np.zeros((3, 5))

""" Set hound position with value 1"""
board[0][1] = 1
board[1][0] = 1
board[2][1] = 1

""" Set hare position with value 2 """
board[1][4] = 2

def print_game_state(board):
    """
    print board
    with X for hounds
    with 0 for hare
    """
    cannot_move = [(0, 0), (2, 0), (0, 4), (2, 4)]
    for i in range(board.shape[0]):
        buffer = ''
        for j in range(board.shape[1]):
            if board[i][j] == 1:
                buffer += 'X\t'
            elif board[i][j] == 2:
                buffer += '0\t'
            elif (i, j) in cannot_move:
                buffer += ' \t'
            else:
                buffer += '-\t'
        print(buffer)
    print('-' * 30)


def get_hare_position(board):
    hare = np.where(board == 2)
    row = hare[0][0]
    col = hare[1][0]
    return (row, col)


def get_hound_positions(board):
    hounds = np.where(board == 1)
    row_hound_1 = hounds[0][0]
    col_hound_1 = hounds[1][0]
    row_hound_2 = hounds[0][1]
    col_hound_2 = hounds[1][1]
    row_hound_3 = hounds[0][2]
    col_hound_3 = hounds[1][2]
    return ((row_hound_1, col_hound_1), (row_hound_2, col_hound_2), (row_hound_3, col_hound_3))


def is_legal_move(player, row_from, col_from, row_to, col_to):
    """
    set player at valid position
    if invalid return False
    """
    illegal_moves = [(0, 0), (2, 0), (0, 4), (2, 4)]

    """special moves that are move available according to diagram
    List of tuples to and from values that are not possible"""
    moves_not_permitted = [[(0, 2), (1, 1)], [(0, 2), (1, 3)], [(1, 1), (2, 2)], [(1, 3), (2, 2)]]
    row_diff = abs(row_from - row_to)
    col_diff = abs(col_from - col_to)

    if player == 'hounds':

        if (row_to >= 0 and row_to < 3 and col_to >= 0 and col_to < 5):
            """Check if the move is not out of bounds for the board with max col range 4 and row range 3
            and then check if it is a legal move"""

            if board[row_to][col_to] == 0 and (row_to, col_to) not in illegal_moves and row_diff <= 1 and col_diff <= 1:
                """ Check if the position is blank.
                Then check if the move is not one of the blank places
                Then check if the row difference and column difference isn't more than 1
                """
                if (col_to - col_from) < 0:  # no moves to the left of the board
                    return False

                for item in moves_not_permitted:
                    if len(set([(row_from, col_from), (row_to, col_to)]).intersection(set(item))) == 2:
                        """ If to and from co-ordinates are present in the moves_not_permitted list then return False"""
                        return False
                    else:
                        pass
                return True
        else:
            return False

    else:
        """When player is a hare"""

        if (row_to >= 0 and row_to < 3 and col_to >= 0 and col_to < 5):
            """Check if the move is not out of bounds for the board with max col range 4 and row range 3
            and then check if it is a legal move"""

            if board[row_to][col_to] == 0 and (row_to, col_to) not in illegal_moves and row_diff <= 1 and col_diff <= 1:
                """ Check if the position is blank.
                Then check if the move is not one of the blank places
                Then check if the row difference and column difference isn't more than 1"""

                for item in moves_not_permitted:
                    if len(set([(row_from, col_from), (row_to, col_to)]).intersection(set(item))) == 2:
                        """ If to and from co-ordinates are present in the moves_not_permitted list then return False"""
                        return False
                    else:
                        pass
                return True

        else:
            return False


def possible_moves_list(row, col):
    """ give the row and col
    return all the possible moves in the matrix"""
    top = (row - 1, col)
    bot = (row + 1, col)
    left = (row, col - 1)
    right = (row, col + 1)
    diagonal_top_left = (row - 1, col - 1)
    diagonal_top_right = (row - 1, col + 1)
    diagonal_bot_left = (row + 1, col - 1)
    diagonal_bot_right = (row + 1, col + 1)
    moves = [top, bot, left, right, diagonal_top_left, diagonal_top_right, diagonal_bot_left, diagonal_bot_right]
    return moves


def get_next_moves(board, player):
    """ List of applicable moves to the board"""

    if player == 'hare':
        moves = []
        next_moves = []

        (row_from, col_from) = get_hare_position(board)
        moves = possible_moves_list(row_from, col_from)

        for move in moves:
            row_to = move[0]
            col_to = move[1]

            if is_legal_move(player, row_from, col_from, row_to, col_to):
                """ if move is allowed then add to list of next moves"""
                next_moves.append(move)

        return next_moves

    else:
        """ for individual hounds
        get next moves"""
        moves = []
        next_moves_hound1 = []
        next_moves_hound2 = []
        next_moves_hound3 = []

        (row_hound_1, col_hound_1), (row_hound_2, col_hound_2), (row_hound_3, col_hound_3) = get_hound_positions(board)
        moves_hound1 = possible_moves_list(row_hound_1, col_hound_1)
        moves_hound2 = possible_moves_list(row_hound_2, col_hound_2)
        moves_hound3 = possible_moves_list(row_hound_3, col_hound_3)

        for move in moves_hound1:
            row_to = move[0]
            col_to = move[1]

            if is_legal_move(player, row_hound_1, col_hound_1, row_to, col_to):
                next_moves_hound1.append(move)

        for move in moves_hound2:
            row_to = move[0]
            col_to = move[1]

            if is_legal_move(player, row_hound_2, col_hound_2, row_to, col_to):
                next_moves_hound2.append(move)

        for move in moves_hound3:
            row_to = move[0]
            col_to = move[1]

            if is_legal_move(player, row_hound_3, col_hound_3, row_to, col_to):
                next_moves_hound3.append(move)

        return (next_moves_hound1, next_moves_hound2, next_moves_hound3)


def evaluation(board, role):
    """ calculate the heuristic value for the given board"""
    start = np.array((1, 4))
    goal = np.array((1, 0))
    (row_hound_1, col_hound_1), (row_hound_2, col_hound_2), (row_hound_3, col_hound_3) = get_hound_positions(board)
    (row_hare, col_hare) = get_hare_position(board)
    hare = np.array((row_hare, col_hare))
    hound1 = np.array((row_hound_1, col_hound_1))
    hound2 = np.array((row_hound_2, col_hound_2))
    hound3 = np.array((row_hound_3, col_hound_3))

    """ Calculate Manhattan distance from the hounds, goal and start state"""
    dist_hare_hound1 = cityblock(hare, hound1)
    dist_hare_hound2 = cityblock(hare, hound2)
    dist_hare_hound3 = cityblock(hare, hound3)
    dist_hare_goal = cityblock(hare, goal)
    dist_hare_start = cityblock(hare, start)

    if role == 'hare':
        score = 0
        """ Check if hare is to the left of the hounds"""
        if (col_hare < col_hound_3 and col_hare < col_hound_2 and col_hare < col_hound_1):
            print(col_hare, col_hound_1, col_hound_2, col_hound_3)
            print(board)
            score = math.pow(10, 5)
        else:
            score = dist_hare_goal - dist_hare_start
            if col_hare > col_hound_1:
                score += dist_hare_hound1 - 1
            if col_hare > col_hound_2:
                score += dist_hare_hound2 - 1
            if col_hare > col_hound_3:
                score += dist_hare_hound3 - 1
            if col_hare == col_hound_1:
                score += dist_hare_hound1
            if col_hare == col_hound_2:
                score += dist_hare_hound2
            if col_hare == col_hound_3:
                score += dist_hare_hound3
        return score
    else:
        score = 0
        """ Check if hare is trapped"""
        if (dist_hare_hound1 == 1 and dist_hare_hound2 == 1 and dist_hare_hound3 == 1):
            score = math.pow(10, 5)
        else:
            score = (dist_hare_hound3 + dist_hare_hound2 + dist_hare_hound1) / 3
            if (1,2) in [(row_hound_1, col_hound_1), (row_hound_2, col_hound_2), (row_hound_3, col_hound_3)]:
                score += 10
            if col_hare > col_hound_1:  # hare is to right on hound1
                score += dist_hare_goal - dist_hare_hound1 - 1
            if col_hare > col_hound_2:  # hare is to right on hound2
                score += dist_hare_goal - dist_hare_hound2 - 1
            if col_hare > col_hound_3:  # hare is to right on hound3
                score += dist_hare_goal - dist_hare_hound3 - 1
            if col_hare == col_hound_1:  # hare is on the same axes as hound1
                score += dist_hare_hound1
            if col_hare == col_hound_2:  # hare is on the same axes as hound2
                score += dist_hare_hound2
            if col_hare == col_hound_3:  # hare is on the same axes as hound3
                score += dist_hare_hound3
        return score


def alternate_player(player):
    if player == 'hare':
        alt_player = 'hounds'
    else:
        alt_player = 'hare'
    return alt_player


def generate_children(board, moves, player, row_hound=None, col_hound=None):
    children = []
    for move in moves:
        copy_board = board.copy()

        if player == 'hare':
            find_hare = np.where(board == 2)
            copy_board[find_hare] = 0
            if copy_board[move[0]][move[1]] == 0:
                copy_board[move[0]][move[1]] = 2
                children.append(copy_board)

        else:
            copy_board[row_hound][col_hound] = 0
            if copy_board[move[0]][move[1]] == 0:
                copy_board[move[0]][move[1]] = 1
                children.append(copy_board)

    return children


def is_gameover(board, score):
    if score >= math.pow(10, 5) or score <= -math.pow(10, 5):
        return True
    else:
        return False


def minimax(board, depth, alpha, beta, player, maximizing_player):

    if depth == 0:
        return (board, evaluation(board, player))

    if maximizing_player:
        """ max player
        get next moves
        select the best move from the child"""
        best_move = None
        children = []

        if player == 'hare':
            """Hare"""
            next_moves = get_next_moves(board, player)
            if next_moves:
                children_hare = generate_children(board, next_moves, player)
                if children_hare:
                    children = children_hare

        else:
            """Hound"""
            next_moves_hound1, next_moves_hound2, next_moves_hound3 = get_next_moves(board, player)
            (row_hound_1, col_hound_1), (row_hound_2, col_hound_2), (row_hound_3, col_hound_3) = get_hound_positions(board)
            if next_moves_hound1:
                children_hound1 = generate_children(board, next_moves_hound1, player, row_hound_1, col_hound_1)
                if children_hound1:
                    children += children + children_hound1
            if next_moves_hound2:
                children_hound2 = generate_children(board, next_moves_hound2, player, row_hound_2, col_hound_2)
                if children_hound2:
                    children += children + children_hound2
            if next_moves_hound3:
                children_hound3 = generate_children(board, next_moves_hound3, player, row_hound_3, col_hound_3)
                if children_hound3:
                    children += children + children_hound3

        for child in children:
            score = minimax(child, depth - 1, alpha, beta, alternate_player(player), False)[1]
            if score > alpha:
                best_move = child
                alpha = score

            if beta <= alpha:
                break  # beta pruning

        return (best_move, alpha)

    else:
        """ min player
        get next moves
        select the best move from the child"""
        best_move = None
        children = []

        if player == 'hare':
            """Hare"""
            next_moves = get_next_moves(board, player)
            if next_moves:
                children_hare = generate_children(board, next_moves, player)
                if children_hare:
                    children = children_hare

        else:
            """Hound"""
            next_moves_hound1, next_moves_hound2, next_moves_hound3 = get_next_moves(board, player)
            (row_hound_1, col_hound_1), (row_hound_2, col_hound_2), (row_hound_3, col_hound_3) = get_hound_positions(board)
            if next_moves_hound1:
                children_hound1 = generate_children(board, next_moves_hound1, player, row_hound_1, col_hound_1)
                if children_hound1:
                    children += children + children_hound1
            if next_moves_hound2:
                children_hound2 = generate_children(board, next_moves_hound2, player, row_hound_2, col_hound_2)
                if children_hound2:
                    children += children + children_hound2
            if next_moves_hound3:
                children_hound3 = generate_children(board, next_moves_hound3, player, row_hound_3, col_hound_3)
                if children_hound3:
                    children += children + children_hound3

        for child in children:
            score = minimax(child, depth - 1, alpha, beta, alternate_player(player), True)[1]

            if score < beta:
                best_move = child
                beta = score

            if beta <= alpha:
                break  # alpha pruning

        return (best_move, beta)

# Main game function
gameover = False
print_game_state(board)
for turn in range(1, 60):
    if turn % 2 == 0:
        player = 'hare'
    else:
        player = 'hounds'
    depth = 7
    alpha = -100000
    beta = 100000
    maximizing_player = True
    best_move, score = minimax(board, depth, alpha, beta, player, maximizing_player)
    board = best_move
    if is_gameover(board, score):
        break
    print(score)

    print_game_state(board)
    gameover = is_gameover(board, score)

print(player + ' WINS!')
