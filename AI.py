import argparse
import random
import os
import pickle
import torch
from deap import base, creator, tools
from minesweeper import Minesweeper

# ----- Agent decision function (works for any height/width) -----
def logic_agent_decide(state, height, width, genome):
    rule1_weight = genome[0]
    rule2_weight = genome[1]
    fallback_weights = genome[2:]

    board = state['board']
    revealed = state['revealed']
    flagged = state['flagged']

    move_candidates = []

    for r in range(height):
        for c in range(width):
            if (r, c) not in revealed or board[r][c] == 0 or board[r][c] == -1:
                continue
            neighbors = [(nr, nc) for nr in range(max(0, r-1), min(height, r+2))
                                    for nc in range(max(0, c-1), min(width, c+2))
                                    if (nr, nc) != (r, c)]
            unrevealed = [pos for pos in neighbors if pos not in revealed and pos not in flagged]
            flagged_n = [pos for pos in neighbors if pos in flagged]

            # Rule 1: Reveal safe neighbors
            if len(flagged_n) == board[r][c] and len(unrevealed) > 0:
                for pos in unrevealed:
                    score = rule1_weight + fallback_weights[pos[0] * width + pos[1]]
                    move_candidates.append((score, ('reveal', pos[0], pos[1])))

            # Rule 2: Flag mines
            if len(unrevealed) + len(flagged_n) == board[r][c] and len(unrevealed) > 0:
                for pos in unrevealed:
                    score = rule2_weight + fallback_weights[pos[0] * width + pos[1]]
                    move_candidates.append((score, ('flag', pos[0], pos[1])))

    # Fallback: reveal cell with max fallback score if no logic moves
    for r in range(height):
        for c in range(width):
            if (r, c) not in revealed and (r, c) not in flagged:
                score = fallback_weights[r * width + c]
                move_candidates.append((score, ('reveal', r, c)))

    if move_candidates:
        best_move = max(move_candidates, key=lambda x: x[0])[1]
        return best_move
    return None

# ----- Evaluation function -----
def evaluate_genome(genome, height=8, width=8, num_mines=10, max_steps=200, device='cpu'):
    # The Minesweeper environment is not vectorized, so we accelerate only batch evaluation
    env = Minesweeper(height=height, width=width, num_mines=num_mines)
    steps = 0
    while not env.game_over and steps < max_steps:
        state = env.get_state()
        move = logic_agent_decide(state, height, width, genome)
        if move is None:
            break
        action, row, col = move
        if action == 'reveal':
            env.reveal(row, col)
        elif action == 'flag':
            env.flag(row, col)
        steps += 1
    fitness = len(env.revealed)
    if env.win:
        fitness += 100
    return (fitness,)

# ----- CUDA batch evaluation -----
def batch_evaluate_genomes(genomes, height, width, num_mines, max_steps, device='cpu'):
    # On CUDA, run in parallel using torch vectorization if possible
    # However, as Minesweeper is not vectorized, we use parallel for-loop with torch if CUDA
    # For real speed-up, proper env vectorization is needed
    results = []
    if device == 'cuda':
        import concurrent.futures
        # Use thread pool as Minesweeper is not GIL-bound
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_genome, genome, height, width, num_mines, max_steps) for genome in genomes]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        # Restore order
        results = [f.result() for f in futures]
    else:
        for genome in genomes:
            results.append(evaluate_genome(genome, height, width, num_mines, max_steps))
    return results

# ----- Save/load functions -----
def save_model(genome, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = "best_genome.pkl"
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(genome, f)

def load_model(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# ----- Evaluation of a single agent -----
def evaluate_agent(genome, height, width, num_mines, test_games, max_steps, verbose=True):
    win_count = 0
    total_revealed = 0
    for i in range(test_games):
        env = Minesweeper(height=height, width=width, num_mines=num_mines)
        steps = 0
        while not env.game_over and steps < max_steps:
            state = env.get_state()
            move = logic_agent_decide(state, height, width, genome)
            if move is None:
                break
            action, row, col = move
            if action == 'reveal':
                env.reveal(row, col)
            elif action == 'flag':
                env.flag(row, col)
            steps += 1
        total_revealed += len(env.revealed)
        if env.win:
            win_count += 1
    if verbose:
        print(f"Agent results over {test_games} games:")
        print(f"  Win rate: {win_count/test_games:.2%}")
        print(f"  Avg cells revealed: {total_revealed/test_games:.2f}")
    return {
        "win_rate": win_count/test_games,
        "avg_revealed": total_revealed/test_games
    }

# ----- Training -----
def train(
    height=8,
    width=8,
    num_mines=10,
    population_size=100,
    generations=40,
    cxpb=0.9,
    mutpb=0.5,
    test_games=20,
    max_steps=200,
    verbose=True,
    save_dir="models",
    device='cpu'
):
    genome_length = 2 + height * width  # 2 rule weights + fallback weights

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=genome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Main training loop
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda fits: sum(fits) / len(fits))
    stats.register("max", max)

    for gen in range(generations):
        # Evaluate all individuals
        if device == 'cuda':
            fitnesses = batch_evaluate_genomes(pop, height, width, num_mines, max_steps, device)
        else:
            fitnesses = [evaluate_genome(ind, height, width, num_mines, max_steps) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        record = stats.compile(pop)
        if verbose:
            print(f"Gen {gen}: Max {record['max']} | Avg {record['avg']:.2f}")
            # fitness_list = [ind.fitness.values[0] for ind in pop]
            # print(f"  Fitnesses: {fitness_list}")
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        pop[:] = offspring

    # Save after all generations
    save_model(hof[0], save_dir)
    print(f"\nFinal best genome saved to {os.path.join(save_dir, 'best_genome.pkl')}")
    print("Best genome (first 20 weights):", [round(x, 3) for x in hof[0][:20]])
    return hof[0]

# ----- Argument parser -----
def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a logic-based genetic agent for Minesweeper (rectangular, CUDA-accelerated)")
    parser.add_argument("--height", type=int, default=16, help="Board height (default: 16)")
    parser.add_argument("--width", type=int, default=30, help="Board width (default: 30)")
    parser.add_argument("--num-mines", type=int, default=99, help="Number of mines (default: 99)")
    parser.add_argument("--population-size", type=int, default=100, help="Population size (default: 100)")
    parser.add_argument("--generations", type=int, default=40, help="Generations (default: 40)")
    parser.add_argument("--cxpb", type=float, default=0.9, help="Crossover prob (default: 0.9)")
    parser.add_argument("--mutpb", type=float, default=0.5, help="Mutation prob (default: 0.5)")
    parser.add_argument("--test-games", type=int, default=20, help="Evaluation games (default: 20)")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per game (default: 200)")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose output")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models (default: models)")
    parser.add_argument("--load-model", type=str, help="Path to a saved model (for evaluation)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the loaded model, do not train")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'auto'], default='auto', help="Device: cpu/cuda/auto (default: auto)")
    args = parser.parse_args()

    # Device selection
    if args.device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    verbose = not args.no_verbose

    if args.load_model:
        print(f"Loading model from {args.load_model} ...")
        genome = load_model(args.load_model)
        if not args.eval_only:
            print("Continuing training from loaded model is not implemented in this script.")
        evaluate_agent(
            genome,
            height=args.height,
            width=args.width,
            num_mines=args.num_mines,
            test_games=args.test_games,
            max_steps=args.max_steps,
            verbose=True
        )
    else:
        best = train(
            height=args.height,
            width=args.width,
            num_mines=args.num_mines,
            population_size=args.population_size,
            generations=args.generations,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            test_games=args.test_games,
            max_steps=args.max_steps,
            verbose=verbose,
            save_dir=args.save_dir,
            device=device
        )
        evaluate_agent(
            best,
            height=args.height,
            width=args.width,
            num_mines=args.num_mines,
            test_games=args.test_games,
            max_steps=args.max_steps,
            verbose=True
        )

if __name__ == "__main__":
    main()
