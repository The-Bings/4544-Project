import numpy as np
import torch
import time
import sys
import argparse
from minesweeper import Minesweeper

# --------- AGENT NETWORK DEFINITION ---------
class AgentNet:
    def __init__(self, genome: np.ndarray, hidden_dim=16):
        # genome: 1D numpy array, length = 3*hidden_dim + hidden_dim + hidden_dim*2 + 2
        self.hidden_dim = hidden_dim
        g = torch.tensor(genome, dtype=torch.float32)
        idx = 0
        self.W1 = g[idx:idx+3*hidden_dim].reshape(3, hidden_dim)
        idx += 3*hidden_dim
        self.b1 = g[idx:idx+hidden_dim]
        idx += hidden_dim
        self.W2 = g[idx:idx+hidden_dim*2].reshape(hidden_dim, 2)
        idx += hidden_dim*2
        self.b2 = g[idx:idx+2]

    def forward(self, obs):
        # obs: [N, 3] tensor
        x = obs @ self.W1 + self.b1
        x = torch.relu(x)
        x = x @ self.W2 + self.b2
        return x # [N, 2] (reveal, flag scores)

    def act(self, obs, mask_reveal, mask_flag):
        # obs: [N, 3]
        logits = self.forward(obs)              # [N, 2]
        # Set invalid actions to -inf
        reveal_scores = logits[:,0].clone()
        flag_scores = logits[:,1].clone()
        reveal_scores[~mask_reveal] = float('-inf')
        flag_scores[~mask_flag] = float('-inf')
        # Find best action
        all_scores = torch.stack([reveal_scores, flag_scores], dim=1) # [N, 2]
        flat_idx = torch.argmax(all_scores)           # index in N*2
        cell_idx = flat_idx // 2
        action_type = flat_idx % 2                    # 0: reveal, 1: flag
        return cell_idx.item(), action_type.item()

# --------- AGENT INPUT ENCODING ---------
def get_agent_obs(state):
    h, w = state['height'], state['width']
    obs = []
    mask_reveal = []
    mask_flag = []
    for r in range(h):
        for c in range(w):
            if (r, c) in state['revealed']:
                clue = state['board'][r][c] / 8.0  # normalized 0..1, mines as -1/8
                revealed = 1.0
                flagged = 1.0 if (r, c) in state['flagged'] else 0.0
            else:
                clue = 0.0
                revealed = 0.0
                flagged = 1.0 if (r, c) in state['flagged'] else 0.0

            obs.append([clue, revealed, flagged])
            mask_reveal.append((r, c) not in state['revealed'] and (r, c) not in state['flagged'] and not state['game_over'])
            mask_flag.append((r, c) not in state['revealed'] and not state['game_over'])
    obs = torch.tensor(obs, dtype=torch.float32)
    mask_reveal = torch.tensor(mask_reveal, dtype=torch.bool)
    mask_flag = torch.tensor(mask_flag, dtype=torch.bool)
    return obs, mask_reveal, mask_flag

# --------- DEMO GUI LOOP ---------
def gui_demo(args):
    # Load genome
    genome = np.load(args.genome)
    # Set up environment
    env = Minesweeper(height=args.height, width=args.width, num_mines=args.mines, seed=args.seed)
    agent = AgentNet(genome)

    # Import pygame
    try:
        import pygame
    except ImportError:
        print("pygame is not installed.")
        sys.exit(1)

    # Pygame constants (from your source)
    GRID_HEIGHT = env.height
    GRID_WIDTH = env.width
    CELL_SIZE = 30
    MARGIN = 2
    WINDOW_WIDTH = GRID_WIDTH * (CELL_SIZE + MARGIN) + MARGIN
    WINDOW_HEIGHT = GRID_HEIGHT * (CELL_SIZE + MARGIN) + MARGIN + 50

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
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Minesweeper AI Demo (Step-by-Step)")
    font = pygame.font.Font(None, 24)

    def draw_board():
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x = col * (CELL_SIZE + MARGIN) + MARGIN
                y = row * (CELL_SIZE + MARGIN) + MARGIN
                is_revealed = (row, col) in env.revealed
                is_flagged = (row, col) in env.flagged
                if is_revealed:
                    pygame.draw.rect(screen, WHITE, (x, y, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE))
                    pygame.draw.line(screen, WHITE, (x, y), (x + CELL_SIZE, y), 2)
                    pygame.draw.line(screen, WHITE, (x, y), (x, y + CELL_SIZE), 2)
                    pygame.draw.line(screen, DARK_GRAY, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
                    pygame.draw.line(screen, DARK_GRAY, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), 2)
                if is_revealed:
                    if env.board[row][col] == env.MINE:
                        pygame.draw.circle(screen, BLACK, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)
                    elif env.board[row][col] > 0:
                        number_text = font.render(str(env.board[row][col]), True, NUMBER_COLORS.get(env.board[row][col], BLACK))
                        text_rect = number_text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                        screen.blit(number_text, text_rect)
                elif is_flagged:
                    pygame.draw.polygon(screen, RED, [(x + CELL_SIZE // 2, y + 5),
                                                      (x + CELL_SIZE - 5, y + CELL_SIZE // 3),
                                                      (x + CELL_SIZE // 2, y + CELL_SIZE // 2)])
                    pygame.draw.rect(screen, BLACK, (x + CELL_SIZE // 2 - 1, y + 5, 2, CELL_SIZE - 10))
                if env.game_over and env.board[row][col] == env.MINE and not is_flagged:
                    if (row, col) not in env.revealed:
                        pygame.draw.circle(screen, BLACK, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)
                if env.game_over and is_flagged and env.board[row][col] != env.MINE:
                    pygame.draw.line(screen, RED, (x + 5, y + 5), (x + CELL_SIZE - 5, y + CELL_SIZE - 5), 2)
                    pygame.draw.line(screen, RED, (x + CELL_SIZE - 5, y + 5), (x + 5, y + CELL_SIZE - 5), 2)

    def draw_status():
        font2 = pygame.font.Font(None, 30)
        flag_text = font2.render(f"Mines: {env.num_mines-len(env.flagged)}", True, BLACK)
        screen.blit(flag_text, (10, WINDOW_HEIGHT - 40))
        status_text = None
        if env.game_over:
            if env.win:
                status_text = font2.render("AI Wins!", True, GREEN)
            else:
                status_text = font2.render("AI Lost!", True, RED)
        if status_text:
            screen.blit(status_text, (WINDOW_WIDTH - status_text.get_width() - 10, WINDOW_HEIGHT - 40))

    running = True
    step_mode = True
    last_step_time = 0
    step_delay = args.delay

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = time.time()
        if step_mode and not env.game_over and now - last_step_time > step_delay:
            state = env.get_state()
            obs, mask_reveal, mask_flag = get_agent_obs(state)
            if mask_reveal.any() or mask_flag.any():
                idx, action_type = agent.act(obs, mask_reveal, mask_flag)
                r = idx // env.width
                c = idx % env.width
                if action_type == 0:
                    env.reveal(r, c)
                else:
                    env.flag(r, c)
                last_step_time = now
            else:
                # No valid moves left
                break

        screen.fill(DARK_GRAY)
        draw_board()
        draw_status()
        pygame.display.flip()
        pygame.time.wait(20)

    # Wait a bit before exit
    pygame.time.wait(2000)
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=str, required=True, help="Path to saved genome .npy file")
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--mines", type=int, default=99)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between AI moves (seconds)")
    args = parser.parse_args()

    gui_demo(args)
