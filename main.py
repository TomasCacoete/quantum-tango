import numpy as np
import sympy as sy
import neal

import re

EMPTY = None
SUN   = 1
MOON  = 0

test_cases = [
    {
        "name": "Linkedin 171",
        "board": [
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [MOON , EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, MOON , EMPTY, EMPTY],
            [EMPTY, EMPTY, MOON , EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, MOON ],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        ],
        "equals_coords": [
            [(1,1), (1,2)],
            [(3,4), (3,5)],
            [(4,3), (4,4)],
        ],
        "opposites_coords": [
            [(0,2), (0,3)]
        ]
    },
]

def print_board(test_case, solution=None):

    board = solution if solution is not None else test_case["board"]
    equal_coords = test_case["equals_coords"]
    opposites_coords = test_case["opposites_coords"]
    n_rows = len(board)
    n_cols = len(board[0])

    board_str = ""

    board_str += "┍-"
    for _ in range(n_cols-1):
        board_str += "┬-"
    board_str += "┑\n"

    for row in range(n_rows-1):
        board_str += "|"
        for col in range(n_cols-1):
            if board[row][col] == SUN:
                board_str += "⟡"
            
            elif board[row][col] == MOON:
                board_str += "☾"

            else:
                board_str += " "


            if [(row, col), (row, col+1)] in equal_coords:
                board_str += "="

            elif [(row, col), (row, col+1)] in opposites_coords:
                board_str += "x"

            else:
                board_str += "|"

        if board[row][n_cols-1] == SUN:
            board_str += "⟡|\n"

        elif board[row][n_cols-1] == MOON:
            board_str += "☾|\n"

        else:
            board_str += " |\n"


        board_str += "|"
        for col in range(n_cols-1):
            if [(row, col), (row+1, col)] in equal_coords:
                board_str += "‖"

            elif [(row, col), (row+1, col)] in opposites_coords:
                board_str += "x"

            else:
                board_str += "-"

            board_str += "┿"

        if [(row, n_cols-1), (row+1, n_cols-1)] in equal_coords:
                board_str += "‖|\n"

        elif [(row, n_cols-1), (row+1, n_cols-1)] in opposites_coords:
            board_str += "x|\n"

        else:
            board_str += "-|\n"

    board_str += "|"
    for col in range(n_cols):
        if board[n_rows-1][col] == SUN:
            board_str += "⟡|"

        elif board[n_rows-1][col] == MOON:
            board_str += "☾|"

        else:
            board_str += " |"

    board_str += "\n┕"
    for _ in range(n_cols-1):
        board_str += "-┴"
    board_str += "-┙\n"

    print(board_str)

def row_col_penalty(symbols):
    penalty = (sum(symbols) - int(len(symbols) / 2)) ** 2

    penalty = penalty.expand()
    return penalty.subs({s**2: s for s in symbols})


def equal_penalty(symbol1, symbol2):
    penalty = (symbol1-symbol2)**2

    penalty = penalty.expand()
    return penalty.subs({symbol1**2: symbol1, symbol2**2: symbol2})


def opposite_penalty(symbol1, symbol2):
    penalty = (symbol1+symbol2-1)**2

    penalty = penalty.expand()
    return penalty.subs({symbol1**2: symbol1, symbol2**2: symbol2})


def three_followed_penalty(symbol1, symbol2, symbol3):
    penalty = ((symbol1 + symbol2 + symbol3)*2/3 - 1)**2
    
    penalty = penalty.expand()
    return penalty.subs({symbol1**2: symbol1, symbol2**2: symbol2, symbol3**2: symbol3})


def penalty_encoding(test_case):
    n_rows = len(test_case["board"])
    n_cols = len(test_case["board"][0])
    
    board_symbols = [[sy.Symbol(f"x{r}_{c}") for c in range(n_cols)] for r in range(n_rows)]

    result = 0

    #rows penalty
    for row in board_symbols:
        result += row_col_penalty(row)

    #cols penalty
    for col in zip(*board_symbols):
        result += row_col_penalty(col)

    #equals penalty
    for coords1, coords2 in test_case["equals_coords"]:
        result += equal_penalty(board_symbols[coords1[0]][coords1[1]], board_symbols[coords2[0]][coords2[1]])

    #opposites penalty
    for coords1, coords2 in test_case["opposites_coords"]:
        result += opposite_penalty(board_symbols[coords1[0]][coords1[1]], board_symbols[coords2[0]][coords2[1]])

    #three followed rows penalty
    for row in board_symbols:
        for i in range(len(row) - 3 + 1):
            result += three_followed_penalty(row[i: i + 3][0], row[i: i + 3][1], row[i: i + 3][2])

    #three followed cols penalty
    for col in zip(*board_symbols):
        for i in range(len(row) - 3 + 1):
            result += three_followed_penalty(row[i: i + 3][0], row[i: i + 3][1], row[i: i + 3][2])

    result = result.expand()
    return result.as_independent(*result.free_symbols, as_Add=True)[1] #tirar o termo constante


def get_symbol_coords(symbol):
    match = re.match(r"x(\d+)_(\d+)", str(symbol))
    if match:
        return int(match.group(1)), int(match.group(2))
    

def get_Q_matrix(expanded_penalty, n_cols, n_rows):
    Q = np.zeros((n_cols * n_rows, n_cols * n_rows))
    
    for term in expanded_penalty.args:
        coeff, vars = term.as_coeff_mul()

        if(len(vars) == 1):
            row, col = get_symbol_coords(vars[0])
            index = row*n_rows+col

            Q[index][index] = coeff

        elif(len(vars) == 2): #termos quadráticos
            row1, col1 = get_symbol_coords(vars[0])
            row2, col2 = get_symbol_coords(vars[1])
            index1 = row1*n_rows+col1
            index2 = row2*n_cols+col2
            
            Q[index1][index2] = coeff
            Q[index2][index1] = coeff

    return Q
            

def translate_Q_matrix_format(Q): #DWave uses a dictionary based representation for the matrix
    Q_dict = {}
    size = Q.shape[0]
    for i in range(size):
        for j in range(i, size):  # upper triangle (symmetric)
            if Q[i][j] != 0:
                Q_dict[(i, j)] = Q[i][j]

    return Q_dict


def get_solution_matrix(sample, n_rows, n_cols):
    solution = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for key, value in sample.items():
        row, col = divmod(key, n_cols)
        solution[row][col] = value

    return solution


def solve(test_case):
    n_rows = len(test_cases[0]["board"])
    n_cols = len(test_cases[0]["board"][0])

    penalty = penalty_encoding(test_cases[0])
    Q = get_Q_matrix(penalty, n_rows, n_cols)
    Q_dict = translate_Q_matrix_format(Q)

    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q_dict, num_reads=100)

    response_list = list(response.data(['sample', 'energy']))
    solution = get_solution_matrix(response_list[0][0], n_rows, n_cols)

    print_board(test_case, solution)


if __name__ == "__main__":
    print_board(test_cases[0])
    solve(test_cases[0])
