from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

import json

EMPTY = None
SUN   = 1
MOON  = 0

def get_n_test_cases(driver):
    driver.get("https://www.tangly.org/")

    #Waiting for page to load
    driver.implicitly_wait(5)

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    select = soup.find('select')
    n_test_cases = int(select.find_all('option')[-1].get('value'))

    return n_test_cases

def parse_board(name, board_html):
    data = {
        "name": name
    }

    cells = board_html.find_all('div', class_='relative')
    for cell in cells:
        if cell.get('class') == ['relative', 'z-10']:
            cells.remove(cell)

    board = [[EMPTY for _ in range(6)] for _ in range(6)]
    equals_coords = []
    opposites_coords = []

    for i in range(len(cells)):
        cell = cells[i]
        direct_children = list(cell.children)
        direct_children = [child for child in direct_children if isinstance(child, str) == False]

        #Getting the equals and opposites
        if len(direct_children) == 2:
            row, col = divmod(i, 6)

            if direct_children[1].get_text(strip=True) == '×':
                if 'bottom-0' in direct_children[1].get("class"):
                    opposites_coords.append([[row, col], [row+1, col]])
                
                else:
                    opposites_coords.append([[row, col], [row, col+1]])

            elif direct_children[1].get_text(strip=True) == '=':
                if 'bottom-0' in direct_children[1].get("class"):
                    equals_coords.append([[row, col], [row+1, col]])
                
                else:
                    equals_coords.append([[row, col], [row, col+1]])

        #Getting the Suns and Moons
        img = direct_children[0].find("img")

        if img and img.get("src") == "/moon.png":
            row, col = divmod(i, 6)
            board[row][col] = MOON

        elif img and img.get("src") == "/lemon.png":
            row, col = divmod(i, 6)
            board[row][col] = SUN

    data['board'] = board
    data['equals_coords'] = equals_coords
    data['opposites_coords'] = opposites_coords

    return data


def get_board(page_html):
    return page_html.find_all("div", class_="grid-cols-6")[0]
 

def get_all_boards_info(driver):
    n_test_cases = get_n_test_cases(driver)

    all_boards = []
    for i in range(n_test_cases):
        driver.get(f"https://www.tangly.org/?id={i+1}")

        #Waiting for page to load
        driver.implicitly_wait(0.5)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        board_html = get_board(soup)
        board_info = parse_board(f"Tangly {i+1}", board_html)

        all_boards.append(board_info)

    with open("test_cases.json", "w") as json_file:
        json.dump(all_boards, json_file)
        

if __name__ == "__main__":
    options = Options()
    options.headless = True

    driver = webdriver.Chrome(options=options)

    get_all_boards_info(driver)

    driver.quit()