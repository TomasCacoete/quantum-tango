import sympy as sy

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

def print_board(test_case):

    board = test_case["board"]
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
    penalty = sum(symbols)
    penalty -= int(len(symbols) / 2)
    penalty = penalty**2

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


def penalty_encoding(test_case):
    n_rows = len(test_case["board"])
    n_cols = len(test_case["board"][0])
    
    board_symbols = [[sy.Symbol(f"x{r}{c}") for c in range(n_cols)] for r in range(n_rows)]

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


    print(result.expand())
#dwave-neal - Passamos os coeficientes e resolve o problema

if __name__ == "__main__":
    #print_board(test_cases[0])

    penalty_encoding(test_cases[0])