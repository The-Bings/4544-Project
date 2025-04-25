import pygame
import random
import sys
from typing import List, Tuple, Set

# Initialize pygame
pygame.init()

# Constants
GRID_SIZE = 10
CELL_SIZE = 30
MARGIN = 2
WINDOW_SIZE = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN
NUM_MINES = 15

# Colors
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

# Numbered cell colors
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

def initialize_board(rows: int, cols: int) -> List[List[int]]:
    """Create a new board filled with zeros (no mines)."""
    return [[0 for _ in range(cols)] for _ in range(rows)]

def place_mines(board: List[List[int]], num_mines: int, first_click_pos: Tuple[int, int]) -> List[List[int]]:
    """Place mines on the board, avoiding the first click position."""
    rows, cols = len(board), len(board[0])
    mines_placed = 0
    
    # Create a safe zone around the first click
    safe_cells = set()
    row, col = first_click_pos
    for r in range(max(0, row-1), min(rows, row+2)):
        for c in range(max(0, col-1), min(cols, col+2)):
            safe_cells.add((r, c))
    
    while mines_placed < num_mines:
        row = random.randint(0, rows - 1)
        col = random.randint(0, cols - 1)
        
        if (row, col) not in safe_cells and board[row][col] != -1:
            board[row][col] = -1  # -1 represents a mine
            mines_placed += 1
    
    return board

def calculate_numbers(board: List[List[int]]) -> List[List[int]]:
    """Calculate the number of adjacent mines for each cell."""
    rows, cols = len(board), len(board[0])
    
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == -1:  # Skip mine cells
                continue
            
            mine_count = 0
            for r in range(max(0, row-1), min(rows, row+2)):
                for c in range(max(0, col-1), min(cols, col+2)):
                    if (r, c) != (row, col) and board[r][c] == -1:
                        mine_count += 1
            
            board[row][col] = mine_count
    
    return board

def reveal_cell(row: int, col: int, board: List[List[int]], revealed: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Reveal a cell and, if it's empty, reveal its neighbors."""
    rows, cols = len(board), len(board[0])
    
    if (row, col) in revealed or row < 0 or row >= rows or col < 0 or col >= cols:
        return revealed
    
    revealed.add((row, col))
    
    # If the cell is empty (0), reveal all adjacent cells
    if board[row][col] == 0:
        for r in range(max(0, row-1), min(rows, row+2)):
            for c in range(max(0, col-1), min(cols, col+2)):
                if (r, c) != (row, col):
                    reveal_cell(r, c, board, revealed)
    
    return revealed

def draw_board(screen, board: List[List[int]], revealed: Set[Tuple[int, int]], flagged: Set[Tuple[int, int]], game_over: bool):
    """Draw the game board on the screen."""
    font = pygame.font.Font(None, 24)
    
    for row in range(len(board)):
        for col in range(len(board[0])):
            x = col * (CELL_SIZE + MARGIN) + MARGIN
            y = row * (CELL_SIZE + MARGIN) + MARGIN
            
            is_revealed = (row, col) in revealed
            is_flagged = (row, col) in flagged
            
            # Draw cell background
            if is_revealed:
                pygame.draw.rect(screen, WHITE, (x, y, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE))
                
                # Draw 3D effect for unrevealed cells
                pygame.draw.line(screen, WHITE, (x, y), (x + CELL_SIZE, y), 2)
                pygame.draw.line(screen, WHITE, (x, y), (x, y + CELL_SIZE), 2)
                pygame.draw.line(screen, DARK_GRAY, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
                pygame.draw.line(screen, DARK_GRAY, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), 2)
            
            # Draw cell content
            if is_revealed:
                if board[row][col] == -1:  # Mine
                    pygame.draw.circle(screen, BLACK, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)
                elif board[row][col] > 0:  # Number
                    number_text = font.render(str(board[row][col]), True, NUMBER_COLORS.get(board[row][col], BLACK))
                    text_rect = number_text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                    screen.blit(number_text, text_rect)
            elif is_flagged:
                # Draw flag
                pygame.draw.polygon(screen, RED, [(x + CELL_SIZE // 2, y + 5), 
                                                (x + CELL_SIZE - 5, y + CELL_SIZE // 3),
                                                (x + CELL_SIZE // 2, y + CELL_SIZE // 2)])
                pygame.draw.rect(screen, BLACK, (x + CELL_SIZE // 2 - 1, y + 5, 2, CELL_SIZE - 10))
            
            # Show all mines when game is over
            if game_over and board[row][col] == -1 and not is_flagged:
                if (row, col) not in revealed:
                    pygame.draw.circle(screen, BLACK, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)
            
            # Show incorrect flags
            if game_over and is_flagged and board[row][col] != -1:
                pygame.draw.line(screen, RED, (x + 5, y + 5), (x + CELL_SIZE - 5, y + CELL_SIZE - 5), 2)
                pygame.draw.line(screen, RED, (x + CELL_SIZE - 5, y + 5), (x + 5, y + CELL_SIZE - 5), 2)

def handle_click(pos: Tuple[int, int], board: List[List[int]], revealed: Set[Tuple[int, int]], 
                flagged: Set[Tuple[int, int]], first_move: bool) -> Tuple[bool, Set[Tuple[int, int]], bool]:
    """Handle a left-click on the board."""
    col = pos[0] // (CELL_SIZE + MARGIN)
    row = pos[1] // (CELL_SIZE + MARGIN)
    
    if row >= GRID_SIZE or col >= GRID_SIZE:
        return first_move, revealed, False
    
    if (row, col) in flagged:
        return first_move, revealed, False
    
    if first_move:
        place_mines(board, NUM_MINES, (row, col))
        calculate_numbers(board)
        first_move = False
    
    # Check if clicked on a mine
    if board[row][col] == -1:
        revealed.add((row, col))
        return first_move, revealed, True  # Game over
    
    # Otherwise reveal the cell
    reveal_cell(row, col, board, revealed)
    return first_move, revealed, False

def handle_right_click(pos: Tuple[int, int], revealed: Set[Tuple[int, int]], 
                      flagged: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Handle a right-click (flagging) on the board."""
    col = pos[0] // (CELL_SIZE + MARGIN)
    row = pos[1] // (CELL_SIZE + MARGIN)
    
    if row >= GRID_SIZE or col >= GRID_SIZE:
        return flagged
    
    if (row, col) not in revealed:
        if (row, col) in flagged:
            flagged.remove((row, col))
        else:
            flagged.add((row, col))
    
    return flagged

def check_win(board: List[List[int]], revealed: Set[Tuple[int, int]]) -> bool:
    """Check if the player has won the game."""
    total_cells = len(board) * len(board[0])
    return len(revealed) == total_cells - NUM_MINES

def draw_status(screen, game_over: bool, win: bool, remaining_flags: int):
    """Draw game status information."""
    font = pygame.font.Font(None, 30)
    
    # Mines remaining
    flag_text = font.render(f"Mines: {remaining_flags}", True, BLACK)
    screen.blit(flag_text, (10, WINDOW_SIZE + 10))
    
    # Game status
    status_text = None
    if game_over:
        if win:
            status_text = font.render("You Win!", True, GREEN)
        else:
            status_text = font.render("Game Over", True, RED)
    
    if status_text:
        screen.blit(status_text, (WINDOW_SIZE - status_text.get_width() - 10, WINDOW_SIZE + 10))

def main():
    """Main game function."""
    # Setup the window
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
    pygame.display.set_caption("Minesweeper")
    
    # Game variables
    board = initialize_board(GRID_SIZE, GRID_SIZE)
    revealed = set()
    flagged = set()
    first_move = True
    game_over = False
    win = False
    
    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not game_over:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        first_move, revealed, game_over = handle_click(event.pos, board, revealed, flagged, first_move)
                        if check_win(board, revealed):
                            game_over = True
                            win = True
                    elif event.button == 3:  # Right click
                        flagged = handle_right_click(event.pos, revealed, flagged)
        
        # Clear the screen
        screen.fill(DARK_GRAY)
        
        # Draw the board
        draw_board(screen, board, revealed, flagged, game_over)
        
        # Draw status
        remaining_flags = NUM_MINES - len(flagged)
        draw_status(screen, game_over, win, remaining_flags)
        
        # Update the display
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()