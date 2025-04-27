import argparse
import numpy as np
import torch

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

def flatten_obs(boards, revealed, flagged):
    B, H, W = boards.shape
    obs = torch.stack([
        boards.float() / 8.0,
        revealed.float(),
        flagged.float()
    ], dim=-1)
    return obs.view(B, H*W, 3)

def agent_forward(obs, weights1, bias1, weights2, bias2):
    x = obs @ weights1 + bias1.unsqueeze(1)
    x = torch.relu(x)
    x = x @ weights2 + bias2.unsqueeze(1)
    return x

def nn_agent(boards, revealed, flagged, genomes, H, W):
    B = boards.shape[0]
    obs = flatten_obs(boards, revealed, flagged)
    hidden_dim = 16
    W1 = genomes[:, :3*hidden_dim].reshape(B, 3, hidden_dim)
    b1 = genomes[:, 3*hidden_dim:3*hidden_dim+hidden_dim]
    W2 = genomes[:, 3*hidden_dim+hidden_dim:3*hidden_dim+hidden_dim+hidden_dim*2].reshape(B, hidden_dim, 2)
    b2 = genomes[:, -2:]
    logits = agent_forward(obs, W1, b1, W2, b2)
    mask_reveal = ~(revealed | flagged)
    mask_flag = ~(revealed)
    logits_reveal = logits[:,:,0].masked_fill(~mask_reveal.view(B, H*W), float('-inf'))
    logits_flag = logits[:,:,1].masked_fill(~mask_flag.view(B, H*W), float('-inf'))
    cat_logits = torch.stack([logits_reveal, logits_flag], dim=-1)  # [B, H*W, 2]
    cat_logits_flat = cat_logits.view(B, -1)
    max_vals = cat_logits_flat.max(dim=1)[0].unsqueeze(1)
    is_max = (cat_logits_flat == max_vals)
    action_indices = torch.arange(cat_logits_flat.shape[1], device=cat_logits.device)
    is_reveal = (action_indices % 2 == 0).unsqueeze(0).expand(B, -1)
    prefer_reveal = is_max & is_reveal
    fallback = is_max.float().argmax(dim=1)
    flat_idx = torch.where(prefer_reveal.any(dim=1), prefer_reveal.float().argmax(dim=1), fallback)
    action_type = flat_idx % 2
    cell_idx = flat_idx // 2
    r = cell_idx // W
    c = cell_idx % W
    return r, c, action_type

def check_win(revealed, flagged, boards, num_mines):
    is_mine = (boards == -1)
    all_safe_revealed = revealed.sum(dim=[1,2]) == (boards.numel() // boards.shape[0]) - num_mines
    all_mines_flagged = (flagged & is_mine).sum(dim=[1,2]) == num_mines
    too_many_flags = flagged.sum(dim=[1,2]) > num_mines
    win = (all_safe_revealed | all_mines_flagged) & (~too_many_flags)
    return win, too_many_flags

def run_generation(genomes, H, W, num_mines, max_steps, device):
    pop_size = genomes.shape[0]
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
        r, c, action_type = nn_agent(boards, revealed, flagged, genomes, H, W)
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
                    # Unflag
                    flagged[i, r[i], c[i]] = False
                    unflag_count[i] += 1
                else:
                    # Only allow flag if total flags used < num_mines
                    if flagged[i].sum() < num_mines:
                        flagged[i, r[i], c[i]] = True
                        if boards[i, r[i], c[i]] == -1:
                            correct_flags[i, r[i], c[i]] = True
                        else:
                            incorrect_flags[i, r[i], c[i]] = True
                    else:
                        # Too many flags: count as invalid action, do not place flag
                        invalid_action_count[i] += 1

            # Check win/flag overflow after every move
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

    # Final check for any that finished with a flag action
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

    fitness = (
        wins * 100
        + 5 * correct_flags.view(pop_size, -1).sum(1).float()
        - 5 * incorrect_flags.view(pop_size, -1).sum(1).float()
        - 2 * invalid_action_count.float()
        + 10 * revealed.view(pop_size, -1).sum(1).float()
        - 3 * unflag_count.float()
        - 100 * too_many_flags_vec.float()  # Large penalty for too many flags
    )
    return fitness, wins.cpu().numpy(), revealed.cpu().numpy(), correct_flags.cpu().numpy(), incorrect_flags.cpu().numpy(), invalid_action_count.cpu().numpy(), unflag_count.cpu().numpy(), too_many_flags_vec.cpu().numpy()

def mutate(pop, sigma=0.1):
    return pop + sigma * torch.randn_like(pop)

def crossover(parents):
    idx = torch.randint(0, 2, parents[0].shape, device=parents[0].device).bool()
    child = parents[0].clone()
    child[idx] = parents[1][idx]
    return child

def evolutionary_loop(args):
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    pop_size = args.population_size
    H, W, num_mines, max_steps = args.height, args.width, args.num_mines, args.max_steps
    hidden_dim = 16
    genome_dim = 3*hidden_dim + hidden_dim + hidden_dim*2 + 2
    if getattr(args, "init_from_genome", None):
        print(f"Initializing population from {args.init_from_genome}")
        best_genome = np.load(args.init_from_genome)
        pop = torch.tensor(best_genome, dtype=torch.float32, device=device).unsqueeze(0).repeat(pop_size, 1)
        pop += 0.1 * torch.randn_like(pop)
    else:
        pop = torch.randn(pop_size, genome_dim, device=device) * 0.5
    best_fitness = -float('inf')
    best_genome = None

    for gen in range(args.generations):
        fitness, wins, revealed, correct_flags, incorrect_flags, invalid_actions, unflags, too_many_flags = run_generation(
            pop, H, W, num_mines, max_steps, device
        )
        avg_fitness = fitness.mean().item()
        win_rate = wins.mean()
        fitness_np = fitness.cpu().numpy()
        best_idx = np.argmax(fitness_np)
        if fitness_np[best_idx] > best_fitness:
            best_fitness = fitness_np[best_idx]
            best_genome = pop[best_idx].detach().cpu().numpy()
        print(f"Gen {gen:03d}: Fitness {avg_fitness:.2f} | Win% {win_rate*100:.2f} | Best {fitness_np[best_idx]:.2f} | "
              f"Avg invalid actions {np.mean(invalid_actions):.2f} | Avg unflags {np.mean(unflags):.2f} | "
              f"Avg too many flags: {np.mean(too_many_flags):.3f}")
        idxs = torch.randint(0, pop_size, (pop_size, 2), device=device)
        fit0 = fitness[idxs[:, 0]]
        fit1 = fitness[idxs[:, 1]]
        winners = torch.where(fit0 >= fit1, idxs[:, 0], idxs[:, 1])
        selected = pop[winners]
        children = []
        for i in range(0, pop_size, 2):
            if i+1 >= pop_size:
                children.append(mutate(selected[i]))
            else:
                child1 = crossover([selected[i], selected[i+1]])
                child2 = crossover([selected[i+1], selected[i]])
                children.append(mutate(child1))
                children.append(mutate(child2))
        pop = torch.stack(children)[:pop_size]
    np.save(args.save_path, best_genome)
    print(f"Best genome saved to {args.save_path}")

def evaluate_genome(args):
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    best_genome = np.load(args.eval_model)
    pop_size = args.eval_games
    hidden_dim = 16
    genome_dim = 3*hidden_dim + hidden_dim + hidden_dim*2 + 2
    assert best_genome.shape[0] == genome_dim, "Genome shape mismatch"
    genomes = torch.tensor(best_genome, dtype=torch.float32, device=device).unsqueeze(0).repeat(pop_size, 1)
    fitness, wins, revealed, correct_flags, incorrect_flags, invalid_actions, unflags, too_many_flags = run_generation(
        genomes, args.height, args.width, args.num_mines, args.max_steps, device
    )
    print(f"Evaluated {pop_size} games: Win rate: {wins.mean()*100:.2f}%, Avg revealed: {revealed.sum(axis=(1,2)).mean():.2f}, "
          f"Avg correct flags: {correct_flags.sum(axis=(1,2)).mean():.2f}, Avg incorrect flags: {incorrect_flags.sum(axis=(1,2)).mean():.2f}, "
          f"Avg invalid actions: {np.mean(invalid_actions):.2f}, Avg unflags: {np.mean(unflags):.2f}, Avg too many flags: {np.mean(too_many_flags):.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-mines", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=str, default="best_genome_flagging_nn.npy")
    parser.add_argument("--init-from-genome", type=str, default=None)
    parser.add_argument("--eval-model", type=str, default=None)
    parser.add_argument("--eval-games", type=int, default=1024)
    args = parser.parse_args()

    if args.eval_model:
        evaluate_genome(args)
    else:
        evolutionary_loop(args)

if __name__ == "__main__":
    main()
