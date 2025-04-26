import random
from typing import List, Tuple, Set, Optional

class Minesweeper:
    MINE = -1

    def __init__(self, grid_size=10, num_mines=15, seed=None):
        self.grid_size = grid_size
        self.num_mines = num_mines
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.board = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.revealed = set()
        self.flagged = set()
        self.first_move = True
        self.game_over = False
        self.win = False

    def place_mines(self, first_click_pos: Tuple[int, int]):
        # Place mines avoiding first click and its neighbors
        rows, cols = self.grid_size, self.grid_size
        safe_cells = set()
        fr, fc = first_click_pos
        for r in range(max(0, fr-1), min(rows, fr+2)):
            for c in range(max(0, fc-1), min(cols, fc+2)):
                safe_cells.add((r, c))
        mines_placed = 0
        while mines_placed < self.num_mines:
            row = self.rng.randint(0, rows - 1)
            col = self.rng.randint(0, cols - 1)
            if (row, col) not in safe_cells and self.board[row][col] != self.MINE:
                self.board[row][col] = self.MINE
                mines_placed += 1
        self.calculate_numbers()

    def calculate_numbers(self):
        rows, cols = self.grid_size, self.grid_size
        for row in range(rows):
            for col in range(cols):
                if self.board[row][col] == self.MINE:
                    continue
                count = 0
                for r in range(max(0, row-1), min(rows, row+2)):
                    for c in range(max(0, col-1), min(cols, col+2)):
                        if (r, c) != (row, col) and self.board[r][c] == self.MINE:
                            count += 1
                self.board[row][col] = count

    def reveal(self, row: int, col: int) -> bool:
        """Reveals a cell. Returns True if game over (stepped on a mine), else False."""
        if self.game_over or (row, col) in self.flagged:
            return False
        if self.first_move:
            self.place_mines((row, col))
            self.first_move = False
        if self.board[row][col] == self.MINE:
            self.revealed.add((row, col))
            self.game_over = True
            self.win = False
            return True
        self._reveal_cell(row, col)
        if self.check_win():
            self.game_over = True
            self.win = True
        return False

    def _reveal_cell(self, row: int, col: int):
        if (row, col) in self.revealed or row < 0 or col < 0 or row >= self.grid_size or col >= self.grid_size:
            return
        self.revealed.add((row, col))
        if self.board[row][col] == 0:
            for r in range(max(0, row-1), min(self.grid_size, row+2)):
                for c in range(max(0, col-1), min(self.grid_size, col+2)):
                    if (r, c) != (row, col):
                        self._reveal_cell(r, c)

    def flag(self, row: int, col: int):
        if (row, col) in self.revealed or self.game_over:
            return
        if (row, col) in self.flagged:
            self.flagged.remove((row, col))
        else:
            self.flagged.add((row, col))

    def get_state(self) -> dict:
        """Return all relevant info for AI (board size, revealed, flagged, board data)."""
        return {
            'revealed': set(self.revealed),
            'flagged': set(self.flagged),
            'board': [row[:] for row in self.board],
            'game_over': self.game_over,
            'win': self.win,
            'first_move': self.first_move
        }

    def set_state(self, state: dict):
        """Set the game state (for sim/AI replay)."""
        self.revealed = set(state['revealed'])
        self.flagged = set(state['flagged'])
        self.board = [row[:] for row in state['board']]
        self.game_over = state['game_over']
        self.win = state['win']
        self.first_move = state['first_move']

    def check_win(self) -> bool:
        total_cells = self.grid_size * self.grid_size
        return len(self.revealed) == total_cells - self.num_mines

    # Text interface for debug/AI
    def print_board(self, reveal_all=False):
        for r in range(self.grid_size):
            line = ''
            for c in range(self.grid_size):
                if reveal_all:
                    v = self.board[r][c]
                else:
                    if (r, c) in self.flagged:
                        v = 'F'
                    elif (r, c) not in self.revealed:
                        v = '?'
                    else:
                        v = self.board[r][c]
                if v == self.MINE:
                    char = '*'
                elif v == 0:
                    char = '.'
                elif v == 'F':
                    char = 'F'
                elif v == '?':
                    char = '?'
                else:
                    char = str(v)
                line += f"{char:2}"
            print(line)
        print()

    # Optional: Human play with pygame GUI
    def play_human(self):
        try:
            import pygame
        except ImportError:
            print("pygame is not installed.")
            return

        # --- GUI constants ---
        GRID_SIZE = self.grid_size
        CELL_SIZE = 30
        MARGIN = 2
        WINDOW_SIZE = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN
        NUM_MINES = self.num_mines

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GRAY = (192, 192, 192)
        DARK_GRAY = (128, 128, 128)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        GREEN = (0, 128, 0)
        CYAN = (0, 128, 128)
        PURPLE = (128, 0, 128)
        MAROON = (128, 0, 0)
        YELLOW = (255, 255, 0)
        NUMBER_COLORS = {
            1: BLUE,
            2: GREEN,
            3: RED,
            4: PURPLE,
            5: MAROON,
            6: CYAN,
            7: BLACK,
            8: GRAY
        }

        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
        pygame.display.set_caption("Minesweeper (AI-ready)")

        font = pygame.font.Font(None, 24)
        def draw_board():
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    x = col * (CELL_SIZE + MARGIN) + MARGIN
                    y = row * (CELL_SIZE + MARGIN) + MARGIN
                    is_revealed = (row, col) in self.revealed
                    is_flagged = (row, col) in self.flagged
                    if is_revealed:
                        pygame.draw.rect(screen, WHITE, (x, y, CELL_SIZE, CELL_SIZE))
                    else:
                        pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE))
                        # 3D effect
                        pygame.draw.line(screen, WHITE, (x, y), (x + CELL_SIZE, y), 2)
                        pygame.draw.line(screen, WHITE, (x, y), (x, y + CELL_SIZE), 2)
                        pygame.draw.line(screen, DARK_GRAY, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
                        pygame.draw.line(screen, DARK_GRAY, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), 2)
                    if is_revealed:
                        if self.board[row][col] == self.MINE:
                            pygame.draw.circle(screen, BLACK, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)
                        elif self.board[row][col] > 0:
                            number_text = font.render(str(self.board[row][col]), True, NUMBER_COLORS.get(self.board[row][col], BLACK))
                            text_rect = number_text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                            screen.blit(number_text, text_rect)
                    elif is_flagged:
                        pygame.draw.polygon(screen, RED, [(x + CELL_SIZE // 2, y + 5),
                                                          (x + CELL_SIZE - 5, y + CELL_SIZE // 3),
                                                          (x + CELL_SIZE // 2, y + CELL_SIZE // 2)])
                        pygame.draw.rect(screen, BLACK, (x + CELL_SIZE // 2 - 1, y + 5, 2, CELL_SIZE - 10))
                    # Show all mines when game is over
                    if self.game_over and self.board[row][col] == self.MINE and not is_flagged:
                        if (row, col) not in self.revealed:
                            pygame.draw.circle(screen, BLACK, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)
                    # Show incorrect flags
                    if self.game_over and is_flagged and self.board[row][col] != self.MINE:
                        pygame.draw.line(screen, RED, (x + 5, y + 5), (x + CELL_SIZE - 5, y + CELL_SIZE - 5), 2)
                        pygame.draw.line(screen, RED, (x + CELL_SIZE - 5, y + 5), (x + 5, y + CELL_SIZE - 5), 2)

        def draw_status():
            font2 = pygame.font.Font(None, 30)
            flag_text = font2.render(f"Mines: {NUM_MINES-len(self.flagged)}", True, BLACK)
            screen.blit(flag_text, (10, WINDOW_SIZE + 10))
            status_text = None
            if self.game_over:
                if self.win:
                    status_text = font2.render("You Win!", True, GREEN)
                else:
                    status_text = font2.render("Game Over", True, RED)
            if status_text:
                screen.blit(status_text, (WINDOW_SIZE - status_text.get_width() - 10, WINDOW_SIZE + 10))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if not self.game_over:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        col = event.pos[0] // (CELL_SIZE + MARGIN)
                        row = event.pos[1] // (CELL_SIZE + MARGIN)
                        if row >= GRID_SIZE or col >= GRID_SIZE:
                            continue
                        if event.button == 1:
                            self.reveal(row, col)
                        elif event.button == 3:
                            self.flag(row, col)
            screen.fill(DARK_GRAY)
            draw_board()
            draw_status()
            pygame.display.flip()
        pygame.quit()

if __name__ == "__main__":
    env = Minesweeper(grid_size=8, num_mines=15)
    env.play_human()  
