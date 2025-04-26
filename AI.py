import argparse
import random
import os
import pickle
from deap import base, creator, tools
from minesweeper import Minesweeper

def logic_agent_decide(state, grid_size, genome):
    board = state['board']
    revealed = state['revealed']
    flagged = state['flagged']
    fallback_weights = genome
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) not in revealed or board[r][c] == 0 or board[r][c] == -1:
                continue
            neighbors = [(nr, nc) for nr in range(max(0, r-1), min(grid_size, r+2))
                                    for nc in range(max(0, c-1), min(grid_size, c+2))
                                    if (nr, nc) != (r, c)]
            unrevealed = [pos for pos in neighbors if pos not in revealed and pos not in flagged]
            flagged_n = [pos for pos in neighbors if pos in flagged]
            if len(flagged_n) == board[r][c] and len(unrevealed) > 0:
                for pos in unrevealed:
                    return ('reveal', pos[0], pos[1])
            if len(unrevealed) + len(flagged_n) == board[r][c] and len(unrevealed) > 0:
                for pos in unrevealed:
                    return ('flag', pos[0], pos[1])
    fallback_candidates = [
        ((r, c), fallback_weights[r * grid_size + c])
        for r in range(grid_size) for c in range(grid_size)
        if (r, c) not in revealed and (r, c) not in flagged
    ]
    if fallback_candidates:
        best_cell = max(fallback_candidates, key=lambda x: x[1])[0]
        return ('reveal', best_cell[0], best_cell[1])
    return None

def evaluate_genome(genome, grid_size=8, num_mines=10, max_steps=200):
    env = Minesweeper(grid_size=grid_size, num_mines=num_mines)
    steps = 0
    while not env.game_over and steps < max_steps:
        state = env.get_state()
        move = logic_agent_decide(state, grid_size, genome)
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

def save_model(genome, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"best_genome.pkl"
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(genome, f)

def load_model(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def evaluate_agent(genome, grid_size, num_mines, test_games, max_steps, verbose=True):
    win_count = 0
    total_revealed = 0
    for i in range(test_games):
        env = Minesweeper(grid_size=grid_size, num_mines=num_mines)
        steps = 0
        while not env.game_over and steps < max_steps:
            state = env.get_state()
            move = logic_agent_decide(state, grid_size, genome)
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

def demo_agent(genome, grid_size, num_mines, max_steps):
    env = Minesweeper(grid_size=grid_size, num_mines=num_mines)
    steps = 0
    print("\nDemo game with best agent:")
    env.print_board()
    while not env.game_over and steps < max_steps:
        state = env.get_state()
        move = logic_agent_decide(state, grid_size, genome)
        if move is None:
            break
        action, row, col = move
        if action == 'reveal':
            env.reveal(row, col)
        elif action == 'flag':
            env.flag(row, col)
        env.print_board()
        steps += 1

def train(
    grid_size=8,
    num_mines=10,
    population_size=100,
    generations=40,
    cxpb=0.9,
    mutpb=0.5,
    test_games=20,
    max_steps=200,
    verbose=True,
    save_dir="models"
):
    genome_length = grid_size * grid_size

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=genome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_genome, grid_size=grid_size, num_mines=num_mines, max_steps=max_steps)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda fits: sum(fits) / len(fits))
    stats.register("max", max)

    for gen in range(generations):
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
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

def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a logic-based genetic agent for Minesweeper using DEAP.")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid size (default: 8)")
    parser.add_argument("--num-mines", type=int, default=10, help="Number of mines (default: 10)")
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
    args = parser.parse_args()

    verbose = not args.no_verbose

    if args.load_model:
        print(f"Loading model from {args.load_model} ...")
        genome = load_model(args.load_model)
        if not args.eval_only:
            print("Continuing training from loaded model is not implemented in this script.")
        evaluate_agent(
            genome,
            grid_size=args.grid_size,
            num_mines=args.num_mines,
            test_games=args.test_games,
            max_steps=args.max_steps,
            verbose=True
        )
    else:
        best = train(
            grid_size=args.grid_size,
            num_mines=args.num_mines,
            population_size=args.population_size,
            generations=args.generations,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            test_games=args.test_games,
            max_steps=args.max_steps,
            verbose=verbose,
            save_dir=args.save_dir
        )
        evaluate_agent(
            best,
            grid_size=args.grid_size,
            num_mines=args.num_mines,
            test_games=args.test_games,
            max_steps=args.max_steps,
            verbose=True
        )
        # Uncomment to see a demo game with board printout:
        # demo_agent(
        #     best,
        #     grid_size=args.grid_size,
        #     num_mines=args.num_mines,
        #     max_steps=args.max_steps
        # )

if __name__ == "__main__":
    main()
