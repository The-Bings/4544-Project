import numpy as np
import torch
import time
import sys
import argparse
from minesweeper import Minesweeper  # Must have patched win logic.

# --------- CNN AGENT NETWORK ---------
class CNNAgentNet:
    def __init__(self, genome: np.ndarray, height, width, hidden_dim=16):
        self.hidden_dim = hidden_dim
        self.height = height
        self.width = width
        g = torch.tensor(genome, dtype=torch.float32)
        idx = 0
        self.conv1_w = g[idx:idx+3*3*3*hidden_dim].reshape(hidden_dim, 3, 3, 3)
        idx += 3*3*3*hidden_dim
        self.conv1_b = g[idx:idx+hidden_dim]
        idx += hidden_dim
        self.conv2_w = g[idx:idx+hidden_dim*hidden_dim*3*3].reshape(hidden_dim, hidden_dim, 3, 3)
        idx += hidden_dim*hidden_dim*3*3
        self.conv2_b = g[idx:idx+hidden_dim]
        idx += hidden_dim
        self.out_w = g[idx:idx+hidden_dim*2].reshape(2, hidden_dim, 1, 1)
        idx += hidden_dim*2
        self.out_b = g[idx:idx+2]
        idx += 2

    def forward(self, obs):
        # obs: [1, H, W, 3] tensor
        x = obs.permute(0, 3, 1, 2)  # [1, 3, H, W]
        x = torch.nn.functional.conv2d(x, self.conv1_w, self.conv1_b, padding=1)
        x = torch.relu(x)
        x = torch.nn.functional.conv2d(x, self.conv2_w, self.conv2_b, padding=1)
        x = torch.relu(x)
        x = torch.nn.functional.conv2d(x, self.out_w, self.out_b)
        x = x.permute(0, 2, 3, 1)  # [1, H, W, 2]
        return x[0]  # [H, W, 2]

    def act(self, obs, mask_reveal, mask_flag):
        logits = self.forward(obs)  # [H, W, 2]
        logits_reveal = logits[..., 0].clone()
        logits_flag = logits[..., 1].clone()
        logits_reveal[~mask_reveal] = float('-inf')
        logits_flag[~mask_flag] = float('-inf')
        flat_reveal = logits_reveal.flatten()
        flat_flag = logits_flag.flatten()
        all_scores = torch.stack([flat_reveal, flat_flag], dim=1)  # [N,2]
        flat_idx = torch.argmax(all_scores)
        cell_idx = flat_idx // 2
        action_type = flat_idx % 2
        r = cell_idx // self.width
        c = cell_idx % self.width
        return r.item(), c.item(), action_type.item()

# --------- AGENT INPUT ENCODING ---------
def get_cnn_agent_obs(state, height, width):
    obs = np.zeros((1, height, width, 3), dtype=np.float32)
    mask_reveal = np.zeros((height, width), dtype=bool)
    mask_flag = np.zeros((height, width), dtype=bool)
    for r in range(height):
        for c in range(width):
            if (r, c) in state['revealed']:
                clue = state['board'][r][c] / 8.0  # normalized, mines as -1/8
                revealed = 1.0
                flagged = 1.0 if (r, c) in state['flagged'] else 0.0
            else:
                clue = 0.0
                revealed = 0.0
                flagged = 1.0 if (r, c) in state['flagged'] else 0.0
            obs[0, r, c, 0] = clue
            obs[0, r, c, 1] = revealed
            obs[0, r, c, 2] = flagged

            mask_reveal[r, c] = (r, c) not in state['revealed'] and (r, c) not in state['flagged'] and not state['game_over']
            mask_flag[r, c] = (r, c) not in state['revealed'] and not state['game_over']
    return torch.tensor(obs), torch.tensor(mask_reveal), torch.tensor(mask_flag)

# --------- DEMO GUI LOOP ---------
def gui_demo(args):
    genome = np.load(args.genome)
    win_count = 0
    loss_count = 0

    try:
        import pygame
    except ImportError:
        print("pygame is not installed.")
        sys.exit(1)

    CELL_SIZE = 30
    MARGIN = 2
    WINDOW_WIDTH = args.width * (CELL_SIZE + MARGIN) + MARGIN
    WINDOW_HEIGHT = args.height * (CELL_SIZE + MARGIN) + MARGIN + 100

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
    pygame.display.set_caption("Minesweeper CNN AI Demo (Step-by-Step)")
    font = pygame.font.Font(None, 24)
    font2 = pygame.font.Font(None, 30)

    def new_game():
        env = Minesweeper(height=args.height, width=args.width, num_mines=args.mines, seed=args.seed)
        agent = CNNAgentNet(genome, args.height, args.width)
        return env, agent

    def draw_board(env):
        for row in range(env.height):
            for col in range(env.width):
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

    def draw_status(env, win_count, loss_count):
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
        total_games = win_count + loss_count
        win_rate = (win_count / total_games * 100) if total_games > 0 else 0.0
        stats_text = font2.render(f"Wins: {win_count}  Losses: {loss_count}  Win Rate: {win_rate:.1f}%", True, BLACK)
        screen.blit(stats_text, (10, WINDOW_HEIGHT - 80))

    env, agent = new_game()
    running = True
    step_mode = True
    last_step_time = 0
    step_delay = args.delay
    game_reset_time = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = time.time()

        if env.game_over:
            if game_reset_time is None:
                game_reset_time = now
                if env.win:
                    win_count += 1
                else:
                    loss_count += 1
            elif now - game_reset_time > 1.0:
                env, agent = new_game()
                game_reset_time = None
                last_step_time = now
        else:
            if step_mode and now - last_step_time > step_delay:
                state = env.get_state()
                obs, mask_reveal, mask_flag = get_cnn_agent_obs(state, args.height, args.width)
                if mask_reveal.any() or mask_flag.any():
                    r, c, action_type = agent.act(obs, mask_reveal, mask_flag)
                    if action_type == 0:
                        env.reveal(r, c)
                    else:
                        env.flag(r, c)
                    last_step_time = now

        screen.fill(DARK_GRAY)
        draw_board(env)
        draw_status(env, win_count, loss_count)
        pygame.display.flip()
        pygame.time.wait(20)

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=str, required=True, help="Path to saved CNN genome .npy file")
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--mines", type=int, default=99)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between AI moves (seconds)")
    args = parser.parse_args()

    gui_demo(args)
