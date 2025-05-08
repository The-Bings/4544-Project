import argparse
import numpy as np
import torch
import os

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

class MinesweeperGNN(torch.nn.Module):
    def __init__(self, node_feat_dim=3, hidden_dim=32, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        out = self.out(x)
        return out

def build_board_graph(board, revealed, flagged, device):
    H, W = board.shape
    node_feats = []
    edge_index = [[], []]
    idx = lambda r, c: r * W + c
    for r in range(H):
        for c in range(W):
            f = [
                float(revealed[r, c]),
                float(flagged[r, c]),
                float(board[r, c]) / 8.0 if revealed[r, c] else 0.0
            ]
            node_feats.append(f)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        edge_index[0].append(idx(r, c))
                        edge_index[1].append(idx(nr, nc))
    x = torch.tensor(node_feats, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    return Data(x=x, edge_index=edge_index)

def setup_boards(pop_size, H, W, num_mines, device):
    boards = torch.zeros((pop_size, H, W), dtype=torch.int8, device=device)
    for i in range(pop_size):
        idx = torch.randperm(H*W, device=device)
        mines = idx[:num_mines]
        r = mines // W
        c = mines % W
        boards[i, r, c] = -1
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
    mine_mask = (boards == -1).float().unsqueeze(1)
    neighbor_mines = torch.nn.functional.conv2d(mine_mask, kernel, padding=1)
    boards = torch.where(boards == -1, boards, neighbor_mines.squeeze(1).to(torch.int8))
    return boards

def first_click(boards, revealed, flagged):
    pop_size, H, W = boards.shape
    idxs = torch.randint(0, H*W, (pop_size,), device=boards.device)
    r = idxs // W
    c = idxs % W
    for i in range(pop_size):
        if not flagged[i, r[i], c[i]]:
            revealed[i, r[i], c[i]] = True
    return r, c

def gnn_agent_forward(gnn, boards, revealed, flagged, device):
    B, H, W = boards.shape
    batch_graphs = []
    for i in range(B):
        data = build_board_graph(boards[i], revealed[i], flagged[i], device)
        batch_graphs.append(data)
    batch = Batch.from_data_list(batch_graphs)
    gnn.eval()
    with torch.no_grad():
        logits = gnn(batch)  # [sum_nodes, 2]
    return logits, batch

def gnn_agent(gnn, boards, revealed, flagged, device):
    B, H, W = boards.shape
    out = []
    logits, batch = gnn_agent_forward(gnn, boards, revealed, flagged, device)
    node_ptr = batch.ptr.cpu().numpy()
    for i in range(B):
        start, end = node_ptr[i], node_ptr[i + 1]
        lgi = logits[start:end]
        mask_reveal = ~(revealed[i].flatten() | flagged[i].flatten())
        mask_flag = ~revealed[i].flatten()
        logits_reveal = lgi[:, 0].clone()
        logits_flag = lgi[:, 1].clone()
        logits_reveal[~mask_reveal] = -float('inf')
        logits_flag[~mask_flag] = -float('inf')
        cat_logits = torch.stack([logits_reveal, logits_flag], dim=1)
        max_val = cat_logits.max()
        is_max = (cat_logits == max_val)
        prefer_reveal = is_max[:, 0]
        if prefer_reveal.any():
            flat_idx = prefer_reveal.float().argmax().item()
            action_type = 0
        else:
            fallback = cat_logits.view(-1).argmax()
            flat_idx = (fallback // 2).item()
            action_type = (fallback % 2).item()
        r = int(flat_idx // W)
        c = int(flat_idx % W)
        out.append((r, c, action_type))
    r, c, action_type = zip(*out)
    return (
        torch.tensor(r, device=device),
        torch.tensor(c, device=device),
        torch.tensor(action_type, device=device)
    )

def check_win(revealed, flagged, boards, num_mines):
    is_mine = (boards == -1)
    all_safe_revealed = revealed.sum(dim=[1,2]) == (boards.numel() // boards.shape[0]) - num_mines
    all_mines_flagged = (flagged & is_mine).sum(dim=[1,2]) == num_mines
    too_many_flags = flagged.sum(dim=[1,2]) > num_mines
    win = (all_safe_revealed | all_mines_flagged) & (~too_many_flags)
    return win, too_many_flags

def run_generation(pop_state_dicts, H, W, num_mines, max_steps, device, hidden_dim=32, eval_games=50):
    pop_size = len(pop_state_dicts)
    total_cells = H * W
    num_safe_cells = total_cells - num_mines

    gnn_list = []
    for i in range(pop_size):
        gnn = MinesweeperGNN(node_feat_dim=3, hidden_dim=hidden_dim)
        gnn.load_state_dict(pop_state_dicts[i])
        gnn = gnn.to(device)
        gnn_list.append(gnn)

    fitness_sum = torch.zeros(pop_size, dtype=torch.float32, device=device)
    wins_sum = torch.zeros(pop_size, dtype=torch.float32, device=device)
    revealed_sum = torch.zeros(pop_size, H, W, dtype=torch.float32, device=device)
    correct_flags_sum = torch.zeros(pop_size, H, W, dtype=torch.float32, device=device)
    incorrect_flags_sum = torch.zeros(pop_size, H, W, dtype=torch.float32, device=device)
    invalid_action_count_sum = torch.zeros(pop_size, dtype=torch.float32, device=device)
    unflag_count_sum = torch.zeros(pop_size, dtype=torch.float32, device=device)
    too_many_flags_sum = torch.zeros(pop_size, dtype=torch.float32, device=device)

    for k in range(eval_games):
        boards = setup_boards(pop_size, H, W, num_mines, device)
        revealed = torch.zeros_like(boards, dtype=torch.bool)
        flagged = torch.zeros_like(boards, dtype=torch.bool)
        correct_flags = torch.zeros_like(boards, dtype=torch.bool)
        incorrect_flags = torch.zeros_like(boards, dtype=torch.bool)
        game_over = torch.zeros(pop_size, dtype=torch.bool, device=device)
        wins = torch.zeros(pop_size, dtype=torch.float32, device=device)
        invalid_action_count = torch.zeros(pop_size, dtype=torch.int32, device=device)
        unflag_count = torch.zeros(pop_size, dtype=torch.int32, device=device)
        too_many_flags_vec = torch.zeros(pop_size, dtype=torch.bool, device=device)

        r, c = first_click(boards, revealed, flagged)
        for i in range(pop_size):
            if boards[i, r[i], c[i]] == -1:
                game_over[i] = True

        for step in range(max_steps):
            r_list, c_list, action_type_list = [], [], []
            for i, gnn in enumerate(gnn_list):
                if game_over[i]:
                    r_list.append(r[i])
                    c_list.append(c[i])
                    action_type_list.append(0)
                    continue
                ri, ci, ai = gnn_agent(gnn, boards[i:i+1], revealed[i:i+1], flagged[i:i+1], device)
                r_list.append(int(ri[0].item()))
                c_list.append(int(ci[0].item()))
                action_type_list.append(int(ai[0].item()))
            r = torch.tensor(r_list, device=device)
            c = torch.tensor(c_list, device=device)
            action_type = torch.tensor(action_type_list, device=device)

            for i in range(pop_size):
                if game_over[i]:
                    continue
                if action_type[i] == 0:
                    if revealed[i, r[i], c[i]] or flagged[i, r[i], c[i]]:
                        invalid_action_count[i] += 1
                        continue
                    if boards[i, r[i], c[i]] == -1:
                        game_over[i] = True
                    else:
                        revealed[i, r[i], c[i]] = True
                elif action_type[i] == 1:
                    if revealed[i, r[i], c[i]] or game_over[i]:
                        invalid_action_count[i] += 1
                        continue
                    if flagged[i, r[i], c[i]]:
                        flagged[i, r[i], c[i]] = False
                        unflag_count[i] += 1
                    else:
                        if flagged[i].sum() < num_mines:
                            flagged[i, r[i], c[i]] = True
                            if boards[i, r[i], c[i]] == -1:
                                correct_flags[i, r[i], c[i]] = True
                            else:
                                incorrect_flags[i, r[i], c[i]] = True
                        else:
                            invalid_action_count[i] += 1

                win, too_many_flags = check_win(
                    revealed[i].unsqueeze(0), flagged[i].unsqueeze(0), boards[i].unsqueeze(0), num_mines
                )
                if win[0]:
                    game_over[i] = True
                    wins[i] = 1.0
                elif too_many_flags[0]:
                    game_over[i] = True
                    wins[i] = 0.0
                    too_many_flags_vec[i] = True

            if game_over.all():
                break

        for i in range(pop_size):
            if not game_over[i]:
                win, too_many_flags = check_win(
                    revealed[i].unsqueeze(0), flagged[i].unsqueeze(0), boards[i].unsqueeze(0), num_mines
                )
                if win[0]:
                    game_over[i] = True
                    wins[i] = 1.0
                elif too_many_flags[0]:
                    game_over[i] = True
                    wins[i] = 0.0
                    too_many_flags_vec[i] = True

        safe_cells_revealed = (
            revealed & (boards != -1)
        ).view(pop_size, -1).sum(1).float()
        progress = safe_cells_revealed / num_safe_cells

        mines_flaged = (
            flagged & (boards == -1)
        ).view(pop_size, -1).sum(1).float()

        num_correct_flags = correct_flags.view(pop_size, -1).sum(1).float()
        num_incorrect_flags = incorrect_flags.view(pop_size, -1).sum(1).float()

        fitness = (
            wins * 100
            + 10 * progress
            + 5 * mines_flaged
            + 2 * num_correct_flags
            - 5 * num_incorrect_flags
            - 10 * too_many_flags_vec.float()
            - 4 * invalid_action_count.float()
        )

        fitness_sum += fitness
        wins_sum += wins
        revealed_sum += revealed.float()
        correct_flags_sum += correct_flags.float()
        incorrect_flags_sum += incorrect_flags.float()
        invalid_action_count_sum += invalid_action_count.float()
        unflag_count_sum += unflag_count.float()
        too_many_flags_sum += too_many_flags_vec.float()

    fitness_avg = fitness_sum / eval_games
    wins_avg = wins_sum / eval_games
    revealed_avg = revealed_sum.cpu().numpy() / eval_games
    correct_flags_avg = correct_flags_sum.cpu().numpy() / eval_games
    incorrect_flags_avg = incorrect_flags_sum.cpu().numpy() / eval_games
    invalid_action_avg = invalid_action_count_sum.cpu().numpy() / eval_games
    unflag_count_avg = unflag_count_sum.cpu().numpy() / eval_games
    too_many_flags_avg = too_many_flags_sum.cpu().numpy() / eval_games

    return (
        fitness_avg,
        wins_avg.cpu().numpy(),
        revealed_avg,
        correct_flags_avg,
        incorrect_flags_avg,
        invalid_action_avg,
        unflag_count_avg,
        too_many_flags_avg
    )

def mutate_state_dict(state_dict, sigma=0.15):
    new_state_dict = {}
    for k, v in state_dict.items():
        noise = torch.randn_like(v) * sigma
        new_state_dict[k] = v + noise
    return new_state_dict

def crossover_state_dict(sd1, sd2):
    child = {}
    for k in sd1.keys():
        mask = torch.randint(0, 2, sd1[k].shape, device=sd1[k].device).bool()
        child[k] = torch.where(mask, sd1[k], sd2[k])
    return child

def evolutionary_loop(args):
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    pop_size = args.population_size
    H, W, num_mines, max_steps = args.height, args.width, args.num_mines, args.max_steps
    hidden_dim = 32

    if getattr(args, "init_from_genome", None):
        print(f"Initializing population from {args.init_from_genome}")
        best_state_dict = torch.load(args.init_from_genome, map_location=device)
        population = []
        for i in range(pop_size):
            if i == 0:
                genome = {k: v.clone().to(device) for k, v in best_state_dict.items()}
            else:
                genome = mutate_state_dict(best_state_dict, sigma=0.13)
                genome = {k: v.to(device) for k, v in genome.items()}
            population.append(genome)
    else:
        gnn_template = MinesweeperGNN(node_feat_dim=3, hidden_dim=hidden_dim).to(device)
        template_state_dict = {k: v.clone().detach().to(device) for k, v in gnn_template.state_dict().items()}
        population = []
        for i in range(pop_size):
            genome = mutate_state_dict(template_state_dict, sigma=0.5)
            genome = {k: v.to(device) for k, v in genome.items()}
            population.append(genome)

    best_fitness = -float('inf')
    best_genome = None

    for gen in range(args.generations):
        fitness, wins, revealed, correct_flags, incorrect_flags, invalid_actions, unflags, too_many_flags = run_generation(
            population, H, W, num_mines, max_steps, device, hidden_dim
        )
        avg_fitness = fitness.mean().item()
        win_rate = wins.mean()
        fitness_np = fitness.cpu().numpy()
        best_idx = np.argmax(fitness_np)
        if fitness_np[best_idx] > best_fitness:
            best_fitness = fitness_np[best_idx]
            best_genome = {k: v.cpu() for k, v in population[best_idx].items()}
        print(f"Gen {gen:03d}: Fitness {avg_fitness:.2f} | Win% {win_rate*100:.2f} | Best {fitness_np[best_idx]:.2f} | "
              f"Avg invalid actions {np.mean(invalid_actions):.2f} | Avg unflags {np.mean(unflags):.2f} | "
              f"Avg too many flags: {np.mean(too_many_flags):.3f}")

        idxs = torch.randint(0, pop_size, (pop_size, 2), device=device)
        fit0 = fitness[idxs[:, 0]]
        fit1 = fitness[idxs[:, 1]]
        winners = torch.where(fit0 >= fit1, idxs[:, 0], idxs[:, 1])
        selected = [population[w.item()] for w in winners]

        children = []
        for i in range(0, pop_size, 2):
            if i+1 >= pop_size:
                child = mutate_state_dict(selected[i])
                child = {k: v.to(device) for k, v in child.items()}
                children.append(child)
            else:
                child1 = mutate_state_dict(crossover_state_dict(selected[i], selected[i+1]))
                child2 = mutate_state_dict(crossover_state_dict(selected[i+1], selected[i]))
                child1 = {k: v.to(device) for k, v in child1.items()}
                child2 = {k: v.to(device) for k, v in child2.items()}
                children.append(child1)
                children.append(child2)
        population = children[:pop_size]

    if best_genome is not None:
        torch.save(best_genome, args.save_path)
        print(f"Best genome saved to {args.save_path}")

def evaluate_genome(args):
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    hidden_dim = 32
    best_state_dict = torch.load(args.eval_model, map_location=device)
    pop_size = args.eval_games
    population = []
    for _ in range(pop_size):
        genome = {k: v.clone().to(device) for k, v in best_state_dict.items()}
        population.append(genome)
    fitness, wins, revealed, correct_flags, incorrect_flags, invalid_actions, unflags, too_many_flags = run_generation(
        population, args.height, args.width, args.num_mines, args.max_steps, device, hidden_dim
    )
    print(f"Evaluated {pop_size} games: Win rate: {wins.mean()*100:.2f}%, Avg revealed: {revealed.sum(axis=(1,2)).mean():.2f}, "
          f"Avg correct flags: {correct_flags.sum(axis=(1,2)).mean():.2f}, Avg incorrect flags: {incorrect_flags.sum(axis=(1,2)).mean():.2f}, "
          f"Avg invalid actions: {np.mean(invalid_actions):.2f}, Avg unflags: {np.mean(unflags):.2f}, Avg too many flags: {np.mean(too_many_flags):.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-mines", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=str, default="best_gnn_model.pt")
    parser.add_argument("--init-from-genome", type=str, default=None)
    parser.add_argument("--eval-model", type=str, default=None)
    parser.add_argument("--eval-games", type=int, default=128)
    args = parser.parse_args()

    if args.eval_model:
        evaluate_genome(args)
    else:
        evolutionary_loop(args)

if __name__ == "__main__":
    main()
