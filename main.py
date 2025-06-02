import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import neal
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

import json
import re

YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

EMPTY = None
SUN   = 1
MOON  = 0

test_case_linkedin = {
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
        [[1,1], [1,2]],
        [[3,4], [3,5]],
        [[4,3], [4,4]]
    ],
    "opposites_coords": [
        [[0,2], [0,3]]
    ]
}

test_case_4_4 = {
    "name": "4x4 Test",
    "board": [
        [EMPTY, EMPTY, MOON , EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, MOON , EMPTY, EMPTY]
    ],
    "equals_coords": [
        [[0,1], [0,2]],
        [[1,0], [2,0]]
    ],
    "opposites_coords": [
        [[1,1], [1,2]]
    ]
}

def convert_value(val):
    match val:
        case None:
            return EMPTY
        case 0:
            return MOON
        case 1:
            return SUN


def load_test_cases(filepath):
    with open(filepath, 'r') as f:
        raw_cases = json.load(f)
    
    test_cases = []
    for case in raw_cases:
        converted_board = [
            [convert_value(cell) for cell in row] for row in case["board"]
        ]

        test_cases.append({
            "name": case["name"],
            "board": converted_board,
            "equals_coords": case["equals_coords"],
            "opposites_coords": case["opposites_coords"],
        })
    
    return test_cases

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
                board_str += f"{YELLOW}⟡{RESET}"
            
            elif board[row][col] == MOON:
                board_str += f"{BLUE}☾{RESET}"

            else:
                board_str += " "

            if [[row, col], [row, col+1]] in equal_coords:
                board_str += "="

            elif [[row, col], [row, col+1]] in opposites_coords:
                board_str += "x"

            else:
                board_str += "|"

        if board[row][n_cols-1] == SUN:
            board_str += f"{YELLOW}⟡{RESET}|\n"

        elif board[row][n_cols-1] == MOON:
            board_str += f"{BLUE}☾{RESET}|\n"

        else:
            board_str += " |\n"


        board_str += "|"
        for col in range(n_cols-1):
            if [[row, col], [row+1, col]] in equal_coords:
                board_str += "‖"

            elif [[row, col], [row+1, col]] in opposites_coords:
                board_str += "x"

            else:
                board_str += "-"

            board_str += "┿"

        if [[row, n_cols-1], [row+1, n_cols-1]] in equal_coords:
                board_str += "‖|\n"

        elif [[row, n_cols-1], [row+1, n_cols-1]] in opposites_coords:
            board_str += "x|\n"

        else:
            board_str += "-|\n"

    board_str += "|"
    for col in range(n_cols):
        if board[n_rows-1][col] == SUN:
            board_str += f"{YELLOW}⟡{RESET}|"

        elif board[n_rows-1][col] == MOON:
            board_str += f"{BLUE}☾{RESET}|"

        else:
            board_str += " |"

    board_str += "\n┕"
    for _ in range(n_cols-1):
        board_str += "-┴"
    board_str += "-┙\n"

    print(board_str)


def starting_moon_penalty(symbol):
    penalty = (symbol-0)**2

    penalty = penalty.expand()
    return penalty.subs({symbol**2: symbol})

def starting_sun_penalty(symbol):
    penalty = (symbol-1)**2

    penalty = penalty.expand()
    return penalty.subs({symbol**2: symbol})

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
    board = test_case["board"]
    
    board_symbols = [[sy.Symbol(f"x{r}_{c}") for c in range(n_cols)] for r in range(n_rows)]

    result = 0

    for i in range(n_rows):
        for j in range(n_cols):
            if board[i][j] == SUN:
                result += starting_sun_penalty(board_symbols[i][j])
            
            elif board[i][j] == MOON:
                result += starting_moon_penalty(board_symbols[i][j])

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
        for i in range(len(col) - 3 + 1):
            result += three_followed_penalty(col[i: i + 3][0], col[i: i + 3][1], col[i: i + 3][2])

    result = result.expand()
    return result.as_independent(*result.free_symbols, as_Add=True)[1] #tirar o termo constante


def get_symbol_coords(symbol):
    match = re.match(r"[xs](\d+)_(\d+)", str(symbol))
    if match:
        return int(match.group(1)), int(match.group(2))
    

def get_Q_matrix(expanded_penalty, n_cols, n_rows):
    Q = np.zeros((n_cols * n_rows, n_cols * n_rows))
    
    for term in expanded_penalty.args:
        coeff, vars = term.as_coeff_mul()

        if len(vars) == 1:
            row, col = get_symbol_coords(vars[0])
            index = row*n_rows+col

            Q[index][index] = coeff

        elif len(vars) == 2: #termos quadráticos
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


def get_qubo_solution_matrix(sample, n_rows, n_cols):
    solution = [[EMPTY for _ in range(n_cols)] for _ in range(n_rows)]

    for key, value in sample.items():
        row, col = divmod(key, n_cols)
        solution[row][col] = value

    return solution


def get_qubo_solution(penalty, n_rows, n_cols):
    Q = get_Q_matrix(penalty, n_rows, n_cols)
    Q_dict = translate_Q_matrix_format(Q)

    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q_dict, num_reads=1)

    response_list = list(response.data(['sample', 'energy']))
    return get_qubo_solution_matrix(response_list[0][0], n_rows, n_cols)


def change_symbol_name(symbol):
    return sy.Symbol(symbol.name.replace('x', 's'))


def qubo_to_ising_model_translation(penalty):
    ising = 0

    for term in penalty.args:
        coeff, vars = term.as_coeff_mul()
        
        if len(vars) == 1: #termos lineares
            s = change_symbol_name(vars[0])
            ising += coeff*((s+1)/2)

        elif len(vars) == 2: #termos quadráticos
            s1 = change_symbol_name(vars[0])
            s2 = change_symbol_name(vars[1])
            ising += coeff*((s1+1)/2)*((s2+1)/2)

    ising = ising.expand()
    return ising.as_independent(*ising.free_symbols, as_Add=True)[1]
        

def get_pauli_string(vars, n_rows, n_cols):
    n_qubits = n_rows * n_cols
    pauli_str = ['I'] * n_qubits

    for var in vars:
        row, col = get_symbol_coords(var)
        index = row * n_cols + col
        pauli_str[index] = 'Z'

    return ''.join(pauli_str)
    

def get_cost_hamiltonian(ising_expr, n_rows, n_cols):
    pauli_strings = []
    coefficients = []

    for term in ising_expr.args:
        coeff, vars = term.as_coeff_mul()

        pauli_str = ''.join(reversed(get_pauli_string(vars, n_rows, n_cols)))
    
        pauli_strings.append(pauli_str)
        coefficients.append(float(coeff))

    return SparsePauliOp(pauli_strings, coefficients)


def get_mixer_hamiltonian(n_rows, n_cols):
    pauli_strings = []
    for i in range(n_rows*n_cols):
        term = list('I' * (n_rows * n_cols))
        term[i] = 'X'
        pauli_strings.append(''.join(term))

    return SparsePauliOp(pauli_strings)


def solve(test_case):
    n_rows = len(test_case["board"])
    n_cols = len(test_case["board"][0])

    penalty = penalty_encoding(test_case)

    ising = qubo_to_ising_model_translation(penalty)
    Hc = get_cost_hamiltonian(ising, n_rows, n_cols)
    Hm = get_mixer_hamiltonian(n_rows, n_cols)

    gamma_values = [Parameter(f"g{i}") for i in range(4)]
    alpha_values = [Parameter(f"a{i}") for i in range(4)]

    qubits = range(n_rows*n_cols)
    qc = QuantumCircuit(n_rows*n_cols)

    qc.h(qubits)

    for gamma, alpha in zip(gamma_values, alpha_values):
        qc.append(PauliEvolutionGate(Hc, time=gamma, label="Hc"), qubits)
        qc.append(PauliEvolutionGate(Hm, time=alpha, label="Hm"), qubits)

    #qc.measure_all()

    qc.draw(output="mpl", fold=-1)
    #plt.show()

    estimator = StatevectorEstimator()
    circuit = qc
    observable = Hc
    g_values = list(np.random.randn(4))
    a_values = list(np.random.randn(4))
    params = g_values + a_values

    pub = (circuit, observable, params)
    job = estimator.run([pub])

    result = job.result()[0]
    print("evs: ", result.data.evs)

    cost_history = []

    def cost_function(params, circuit, observable, estimator):
        pub = (circuit, observable, params)
        job = estimator.run([pub])
        result = job.result()[0]
        average_energy = result.data.evs

        cost_history.append(average_energy)
        return average_energy
    
    init_params = np.random.randn(2*4)

    optimization_result = minimize(
        cost_function,
        init_params,
        args=(qc, Hc, estimator),
        method="COBYLA",
        tol=0.01
    )

    print("result: ", optimization_result)

    fig, ax = plt.subplots()
    iterations = range(len(cost_history))
    ax.plot(iterations, cost_history)

    # plt.show()

    circuit = qc.copy()
    circuit.measure_all()
    optimal_params = optimization_result.x

    sampler = StatevectorSampler()
    pub = (circuit, optimal_params)
    job = sampler.run([pub], shots=1000)

    result = job.result()[0]
    counts = result.data.meas.get_counts()
    plot_histogram(counts)

    plt.show()

    # QUBO Solution
    # print_board(test_case, get_qubo_solution(penalty, n_rows, n_cols))


if __name__ == "__main__":
    #test_cases = load_test_cases('test_cases.json')
    #print_board(test_cases[0])

    test_case4_4 = {
        "name": "4x4 Test",
        "board": [
            [EMPTY, EMPTY, MOON , EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, MOON , EMPTY, EMPTY]
        ],
        "equals_coords": [
            [[0,1], [0,2]],
            [[1,0], [2,0]]
        ],
        "opposites_coords": [
            [[1,1], [1,2]]
        ]
    }

    test_case2_2 = {
        "name": "2x2 Test",
        "board": [
            [EMPTY, MOON ],
            [EMPTY, EMPTY]
        ],
        "equals_coords": [],
        "opposites_coords": []
    }

    print_board(test_case2_2)
    solve(test_case2_2)


    #solve(test_cases[0])