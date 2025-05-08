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
    revealed[torch.arange(pop_size), r, c] = True
    return r, c

def cnn_agent_forward(obs, genomes, H, W, hidden=16):
    # obs: [B, H, W, 3]
    B = obs.shape[0]
    idx = 0
    conv1_w = genomes[:, idx:idx+3*3*3*hidden].reshape(B, hidden, 3, 3, 3)
    idx += 3*3*3*hidden
    conv1_b = genomes[:, idx:idx+hidden]  # shape [B, hidden]
    idx += hidden
    conv2_w = genomes[:, idx:idx+hidden*hidden*3*3].reshape(B, hidden, hidden, 3, 3)
    idx += hidden*hidden*3*3
    conv2_b = genomes[:, idx:idx+hidden]  # shape [B, hidden]
    idx += hidden
    out_w = genomes[:, idx:idx+hidden*2].reshape(B, 2, hidden, 1, 1)
    idx += hidden*2
    out_b = genomes[:, idx:idx+2]  # shape [B, 2]

    x = obs.permute(0, 3, 1, 2) # [B, 3, H, W]
    # Conv1
    x = torch.stack([
        torch.nn.functional.conv2d(
            x[i:i+1], conv1_w[i], conv1_b[i], padding=1
        ) for i in range(B)
    ], dim=0).squeeze(1)
    x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    # Conv2
    x = torch.stack([
        torch.nn.functional.conv2d(
            x[i:i+1], conv2_w[i], conv2_b[i], padding=1
        ) for i in range(B)
    ], dim=0).squeeze(1)
    x = torch.nn.functional.leaky_relu(x,negative_slope=0.01)
    # Output 1x1 conv
    x = torch.stack([
        torch.nn.functional.conv2d(
            x[i:i+1], out_w[i], out_b[i]
        ) for i in range(B)
    ], dim=0).squeeze(1)
    x = x.permute(0, 2, 3, 1) # [B, H, W, 2]
    return x

def cnn_agent(boards, revealed, flagged, genomes, H, W, hidden=16):
    B = boards.shape[0]
    obs = torch.stack([
        boards.float() / 8.0,
        revealed.float(),
        flagged.float()
    ], dim=-1)  # [B, H, W, 3]
    logits = cnn_agent_forward(obs, genomes, H, W, hidden=hidden)  # [B, H, W, 2]
    mask_reveal = ~(revealed | flagged)
    mask_flag = ~revealed
    logits_reveal = logits[..., 0].masked_fill(~mask_reveal, float('-inf'))
    logits_flag = logits[..., 1].masked_fill(~mask_flag, float('-inf'))
    cat_logits = torch.stack([logits_reveal, logits_flag], dim=-1)  # [B, H, W, 2]
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
    all_safe_revealed = revealed.sum(dim=[1,2]) == (boards[0].numel() - num_mines)
    all_mines_flagged = (flagged & is_mine).sum(dim=[1,2]) == num_mines
    too_many_flags = flagged.sum(dim=[1,2]) > num_mines
    win = (all_safe_revealed | all_mines_flagged) & (~too_many_flags)
    return win, too_many_flags

def run_generation(genomes, H, W, num_mines, max_steps, device, hidden_dim=16, eval_games=50):
    pop_size = genomes.shape[0]
    total_cells = H * W
    num_safe_cells = total_cells - num_mines

    # Stats accumulators
    fitness_sum = torch.zeros(pop_size, device=device)
    wins_sum = torch.zeros(pop_size, device=device)
    revealed_sum = torch.zeros(pop_size, H, W, device=device)
    correct_flags_sum = torch.zeros(pop_size, H, W, device=device)
    incorrect_flags_sum = torch.zeros(pop_size, H, W, device=device)
    invalid_action_count_sum = torch.zeros(pop_size, device=device)
    unflag_count_sum = torch.zeros(pop_size, device=device)
    too_many_flags_sum = torch.zeros(pop_size, device=device)

    for k in range(eval_games):
        boards = setup_boards(pop_size, H, W, num_mines, device)
        revealed = torch.zeros_like(boards, dtype=torch.bool)
        flagged = torch.zeros_like(boards, dtype=torch.bool)
        correct_flags = torch.zeros_like(boards, dtype=torch.bool)
        incorrect_flags = torch.zeros_like(boards, dtype=torch.bool)
        game_over = torch.zeros(pop_size, dtype=torch.bool, device=device)
        wins = torch.zeros(pop_size, device=device)
        invalid_action_count = torch.zeros(pop_size, device=device)
        unflag_count = torch.zeros(pop_size, device=device)
        too_many_flags_vec = torch.zeros(pop_size, dtype=torch.bool, device=device)

        # First click: safe start everywhere
        r, c = first_click(boards, revealed, flagged)
        # If first click is a mine, game over
        first_mine = boards[torch.arange(pop_size), r, c] == -1
        game_over[first_mine] = True

        for step in range(max_steps):
            # Get actions for all agents
            r, c, action_type = cnn_agent(boards, revealed, flagged, genomes, H, W, hidden=hidden_dim)
            batch_idx = torch.arange(pop_size, device=device)
            not_over = ~game_over

            # Reveal actions
            is_reveal = (action_type == 0) & not_over
            reveal_valid = is_reveal & (~revealed[batch_idx, r, c]) & (~flagged[batch_idx, r, c])
            # Invalid reveal: already revealed or flagged
            invalid_reveal = is_reveal & ~reveal_valid
            invalid_action_count[invalid_reveal] += 1
            # Reveal action: is mine? End game; else reveal cell
            reveal_mine = reveal_valid & (boards[batch_idx, r, c] == -1)
            game_over[reveal_mine] = True
            not_mine = reveal_valid & (~reveal_mine)
            revealed[batch_idx[not_mine], r[not_mine], c[not_mine]] = True

            # Flag/unflag actions
            is_flag = (action_type == 1) & not_over
            can_flag = is_flag & (~revealed[batch_idx, r, c])
            # Invalid flag: on revealed or after game over
            invalid_flag = is_flag & ~can_flag
            invalid_action_count[invalid_flag] += 1

            already_flagged = can_flag & flagged[batch_idx, r, c]
            flagged[batch_idx[already_flagged], r[already_flagged], c[already_flagged]] = False
            unflag_count[already_flagged] += 1

            need_flag = can_flag & (~flagged[batch_idx, r, c])
            # Only flag if flag count < num_mines
            flag_count = flagged.sum(dim=[1,2])
            can_really_flag = need_flag & (flag_count < num_mines)
            flagged[batch_idx[can_really_flag], r[can_really_flag], c[can_really_flag]] = True
            # Correct/incorrect flag bookkeeping
            correct = can_really_flag & (boards[batch_idx, r, c] == -1)
            incorrect = can_really_flag & (boards[batch_idx, r, c] != -1)
            correct_flags[batch_idx[correct], r[correct], c[correct]] = True
            incorrect_flags[batch_idx[incorrect], r[incorrect], c[incorrect]] = True
            # Can't flag: used too many flags
            cant_flag = need_flag & ~(flag_count < num_mines)
            invalid_action_count[cant_flag] += 1

            # Check win or flag overflow for all
            win, too_many_flags = check_win(revealed, flagged, boards, num_mines)
            just_won = win & (~game_over)
            game_over[just_won] = True
            wins[just_won] = 1.0
            just_lost = too_many_flags & (~game_over)
            game_over[just_lost] = True
            wins[just_lost] = 0.0
            too_many_flags_vec[just_lost] = True

            if game_over.all():
                break

        # Final check for any unfinished games
        still_running = ~game_over
        if still_running.any():
            win, too_many_flags = check_win(revealed, flagged, boards, num_mines)
            just_won = win & still_running
            just_lost = too_many_flags & still_running
            game_over[just_won] = True
            wins[just_won] = 1.0
            game_over[just_lost] = True
            wins[just_lost] = 0.0
            too_many_flags_vec[just_lost] = True

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
            wins * 1000
            + 5 * progress
            + 50 * mines_flaged
            + 10 * num_correct_flags
            - 20 * num_incorrect_flags
            - 100 * too_many_flags_vec.float()
            - 5 * invalid_action_count.float()
        )
        # Accumulate stats
        fitness_sum += fitness
        wins_sum += wins
        revealed_sum += revealed.float()
        correct_flags_sum += correct_flags.float()
        incorrect_flags_sum += incorrect_flags.float()
        invalid_action_count_sum += invalid_action_count.float()
        unflag_count_sum += unflag_count.float()
        too_many_flags_sum += too_many_flags_vec.float()

    # Average over eval_games
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
    genome_dim = 3*3*3*hidden_dim + hidden_dim + hidden_dim*hidden_dim*3*3 + hidden_dim + hidden_dim*2 + 2
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
            pop, H, W, num_mines, max_steps, device, hidden_dim
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
    hidden_dim = 16
    genome_dim = 3*3*3*hidden_dim + hidden_dim + hidden_dim*hidden_dim*3*3 + hidden_dim + hidden_dim*2 + 2
    best_genome = np.load(args.eval_model)
    pop_size = args.eval_games
    assert best_genome.shape[0] == genome_dim, "Genome shape mismatch"
    genomes = torch.tensor(best_genome, dtype=torch.float32, device=device).unsqueeze(0).repeat(pop_size, 1)
    fitness, wins, revealed, correct_flags, incorrect_flags, invalid_actions, unflags, too_many_flags = run_generation(
        genomes, args.height, args.width, args.num_mines, args.max_steps, device, hidden_dim
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
    parser.add_argument("--save-path", type=str, default="best_genome_flagging_cnn.npy")
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
