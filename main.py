EMPTY = None
SUN = 1
MOON = 0

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


#dwave-neal - Passamos os coeficientes e resolve o problema

if __name__ == "__main__":
    print_board(test_cases[0])