import argparse
import numpy as np
import torch
from deap import base, creator, tools
from vectorized_minesweeper import VectorizedMinesweeper

def batch_logic_agent_decide(state, height, width, genome, device):
    batch_size = state['board'].shape[0]
    rule1_weight = genome[0]
    rule2_weight = genome[1]
    fallback_weights = torch.tensor(genome[2:], device=device)
    fallback_weights = fallback_weights.view(height, width)
    actions = []
    for i in range(batch_size):
        board = state['board'][i]
        revealed = state['revealed'][i]
        flagged = state['flagged'][i]
        move_candidates = []
        for r in range(height):
            for c in range(width):
                if not revealed[r, c] or board[r, c] == 0 or board[r, c] == -1:
                    continue
                # Neighbors
                unrevealed = []
                flagged_n = []
                for nr in range(max(0, r-1), min(height, r+2)):
                    for nc in range(max(0, c-1), min(width, c+2)):
                        if (nr, nc) == (r, c):
                            continue
                        if not revealed[nr, nc] and not flagged[nr, nc]:
                            unrevealed.append((nr, nc))
                        if flagged[nr, nc]:
                            flagged_n.append((nr, nc))
                if len(flagged_n) == board[r, c] and len(unrevealed) > 0:
                    for pos in unrevealed:
                        score = rule1_weight + fallback_weights[pos]
                        move_candidates.append((score.item(), 'reveal', pos[0], pos[1]))
                if len(unrevealed) + len(flagged_n) == board[r, c] and len(unrevealed) > 0:
                    for pos in unrevealed:
                        score = rule2_weight + fallback_weights[pos]
                        move_candidates.append((score.item(), 'flag', pos[0], pos[1]))
        for r in range(height):
            for c in range(width):
                if not revealed[r, c] and not flagged[r, c]:
                    score = fallback_weights[r, c]
                    move_candidates.append((score.item(), 'reveal', r, c))
        if move_candidates:
            best = max(move_candidates, key=lambda x: x[0])
            actions.append((best[1], best[2], best[3]))
        else:
            actions.append(('none', 0, 0))
    return actions

def batch_evaluate_genomes(genomes, height, width, num_mines, max_steps, device):
    batch_size = len(genomes)
    env = VectorizedMinesweeper(batch_size=batch_size, height=height, width=width, num_mines=num_mines, device=device)
    env.reset()
    first_moves = torch.randint(0, height*width, (batch_size,), device=device)
    r = (first_moves // width).tolist()
    c = (first_moves % width).tolist()
    env.reveal(r, c, indices=list(range(batch_size)))
    steps = 1
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    while active.any() and steps < max_steps:
        state = env.get_state()
        acts = batch_logic_agent_decide(state, height, width, genomes[0], device)
        r_reveal, c_reveal, idx_reveal = [], [], []
        r_flag, c_flag, idx_flag = [], [], []
        for idx, (a, rr, cc) in enumerate(acts):
            if not active[idx]:
                continue
            if a == 'reveal':
                idx_reveal.append(idx)
                r_reveal.append(rr)
                c_reveal.append(cc)
            elif a == 'flag':
                idx_flag.append(idx)
                r_flag.append(rr)
                c_flag.append(cc)
        if idx_reveal:
            env.reveal(r_reveal, c_reveal, indices=idx_reveal)
        if idx_flag:
            env.flag(r_flag, c_flag, indices=idx_flag)
        active = ~env.batch_done()
        steps += 1
    state = env.get_state()
    fitness = (env.batch_won().float() * 100 + state['revealed'].view(batch_size, -1).sum(1)).tolist()
    return [(f,) for f in fitness]

def evaluate_saved_genome(genome_path, height, width, num_mines, games, max_steps, device):
    print(f"Evaluating saved genome from {genome_path}")
    genome = np.load(genome_path)
    batch_size = games
    env = VectorizedMinesweeper(batch_size=batch_size, height=height, width=width, num_mines=num_mines, device=device)
    env.reset()
    first_moves = torch.randint(0, height*width, (batch_size,), device=device)
    r = (first_moves // width).tolist()
    c = (first_moves % width).tolist()
    env.reveal(r, c, indices=list(range(batch_size)))
    steps = 1
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    while active.any() and steps < max_steps:
        state = env.get_state()
        acts = batch_logic_agent_decide(state, height, width, genome, device)
        r_reveal, c_reveal, idx_reveal = [], [], []
        r_flag, c_flag, idx_flag = [], [], []
        for idx, (a, rr, cc) in enumerate(acts):
            if not active[idx]:
                continue
            if a == 'reveal':
                idx_reveal.append(idx)
                r_reveal.append(rr)
                c_reveal.append(cc)
            elif a == 'flag':
                idx_flag.append(idx)
                r_flag.append(rr)
                c_flag.append(cc)
        if idx_reveal:
            env.reveal(r_reveal, c_reveal, indices=idx_reveal)
        if idx_flag:
            env.flag(r_flag, c_flag, indices=idx_flag)
        active = ~env.batch_done()
        steps += 1
    state = env.get_state()
    wins = env.batch_won().sum().item()
    revealed = state['revealed'].sum(dim=(1,2)).cpu().numpy()
    print(f"Evaluation results on {games} games:")
    print(f"Wins: {wins}/{games} ({wins/games*100:.2f}%)")
    print(f"Average revealed cells: {revealed.mean():.2f} / {height*width}")
    return wins, revealed

def main():
    parser = argparse.ArgumentParser(description="Vectorized Minesweeper Evolution (PyTorch/CUDA)")
    parser.add_argument("--height", type=int, default=8, help="Board height")
    parser.add_argument("--width", type=int, default=8, help="Board width")
    parser.add_argument("--num-mines", type=int, default=10, help="Number of mines")
    parser.add_argument("--population-size", type=int, default=128, help="Population size")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per game")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--save-path", type=str, default="best_genome.npy", help="Path to save the best genome")
    parser.add_argument("--eval-model", type=str, default=None, help="Path to a saved genome to evaluate")
    parser.add_argument("--eval-games", type=int, default=256, help="Number of games for evaluation")
    args = parser.parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        if args.device == 'cuda':
            print("Warning: CUDA requested but not available, switching to CPU.")
        device = torch.device('cpu')
    print(f"Using device: {device}")

    height, width = args.height, args.width
    num_mines = args.num_mines
    population_size = args.population_size
    generations = args.generations
    max_steps = args.max_steps

    genome_length = 2 + height * width

    if args.eval_model is not None:
        evaluate_saved_genome(
            args.eval_model,
            height, width,
            num_mines,
            args.eval_games,
            max_steps,
            device
        )
        return

    # DEAP setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: np.random.uniform(-2, 2))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=genome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    for gen in range(generations):
        batch_genomes = [ind for ind in pop]
        batch_genomes_arr = np.array(batch_genomes)
        fitnesses = batch_evaluate_genomes(batch_genomes_arr, height, width, num_mines, max_steps, device)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        print(f"Gen {gen}: Best {hof[0].fitness.values[0]}")
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.9:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if np.random.rand() < 0.5:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        pop[:] = offspring

    print("Best genome:", hof[0])
    np.save(args.save_path, np.array(hof[0], dtype=np.float32))
    print(f"Best genome saved to {args.save_path}")

if __name__ == "__main__":
    main()
