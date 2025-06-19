import random
import time
import heapq
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

class PuzzleSolver:
    def __init__(self):
        self.N = 3
        self.goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.algorithms = ["BFS", "DFS", "IDS", "UCS", "BiDir", "Greedy", "A*", "IDA*"]

    def generate_random_board(self):
        """Generate a random solvable board configuration"""
        while True:
            board = random.sample(range(9), 9)
            if self.is_solvable(board) and not self.is_solved(board):
                return board

    def is_solvable(self, board):
        """Check if the board configuration is solvable"""
        inversions = 0
        for i in range(len(board)):
            for j in range(i+1, len(board)):
                if board[i] and board[j] and board[i] > board[j]:
                    inversions += 1
        return inversions % 2 == 0

    def is_solved(self, board):
        """Check if the board is in the goal state"""
        return board[:8] == list(range(1, 9)) and board[8] == 0

    def get_neighbors(self, state):
        """Get all possible neighboring states"""
        neighbors = []
        empty_idx = state.index(0)
        row, col = empty_idx // self.N, empty_idx % self.N

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.N and 0 <= new_col < self.N:
                swap_idx = new_row * self.N + new_col
                new_state = list(state)
                new_state[empty_idx], new_state[swap_idx] = new_state[swap_idx], new_state[empty_idx]
                neighbors.append(tuple(new_state))
        return neighbors

    def manhattan_distance(self, state):
        """Calculate Manhattan distance heuristic"""
        distance = 0
        for i, value in enumerate(state):
            if value != 0:
                target_row, target_col = (value - 1) // self.N, (value - 1) % self.N
                current_row, current_col = i // self.N, i % self.N
                distance += abs(target_row - current_row) + abs(target_col - current_col)
        return distance

    def reconstruct_path(self, visited, start, goal):
        """Reconstruct the solution path"""
        path = []
        current = goal
        while current != start:
            prev = visited[current]
            path.append(current)
            current = prev
        return len(path)

    def reconstruct_path_ucs(self, visited, start, goal):
        """Reconstruct path for UCS/A*"""
        path_length = 0
        current = goal
        while current != start:
            prev = visited[current][0]
            path_length += 1
            current = prev
        return path_length

    def solve_bfs(self, board):
        """Breadth-First Search"""
        start = tuple(board)
        goal = self.goal

        if start == goal:
            return 0, 1, 0

        start_time = time.time()
        queue = deque([start])
        visited = {start: None}
        states_visited = 1
        max_queue_size = 1

        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            current = queue.popleft()

            if current == goal:
                path_length = self.reconstruct_path(visited, start, goal)
                return path_length, states_visited, max_queue_size

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
                    states_visited += 1

        return -1, states_visited, max_queue_size

    def solve_dfs(self, board):
        """Depth-First Search - Memory efficient version"""
        start = tuple(board)
        goal = self.goal

        if start == goal:
            return 0, 1, 0

        start_time = time.time()
        # Stack stores tuples of (state, path) where path is a list of states from start
        stack = [(start, [start])]
        states_visited = 1
        max_stack_size = 1
        
        # Set a reasonable depth limit to prevent infinite loops
        max_depth = 500

        while stack:
            max_stack_size = max(max_stack_size, len(stack))
            current, path = stack.pop()

            if current == goal:
                return len(path) - 1, states_visited, max_stack_size

            # Only expand if we haven't exceeded depth limit
            if len(path) < max_depth:
                for neighbor in self.get_neighbors(current):
                    # Avoid immediate cycles by checking if neighbor is not in recent path
                    if neighbor not in path[-3:]:  # Avoid last 3 states to prevent immediate cycles
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path))
                        states_visited += 1

        return -1, states_visited, max_stack_size

    def solve_ids(self, board):
        """Iterative Deepening Search"""
        start = tuple(board)
        goal = self.goal

        if start == goal:
            return 0, 1, 0

        total_states = 0
        max_space = 0

        for depth in range(80):
            result, states, space = self.dls(start, goal, depth)
            total_states += states
            max_space = max(max_space, space)
            if result != -1:
                return result, total_states, max_space

        return -1, total_states, max_space

    def dls(self, start, goal, max_depth):
        """Depth-Limited Search helper for IDS"""
        stack = [(start, 0, [start])]
        states_visited = 0
        max_stack_size = 1

        while stack:
            max_stack_size = max(max_stack_size, len(stack))
            current, depth, path = stack.pop()
            states_visited += 1

            if current == goal:
                return len(path) - 1, states_visited, max_stack_size

            if depth < max_depth:
                for neighbor in self.get_neighbors(current):
                    if neighbor not in path:
                        stack.append((neighbor, depth + 1, path + [neighbor]))

        return -1, states_visited, max_stack_size

    def solve_ucs(self, board):
        """Uniform Cost Search"""
        start = tuple(board)
        goal = self.goal

        if start == goal:
            return 0, 1, 0

        heap = [(0, start)]
        visited = {start: (None, 0)}
        states_visited = 1
        max_heap_size = 1

        while heap:
            max_heap_size = max(max_heap_size, len(heap))
            cost, current = heapq.heappop(heap)

            if current == goal:
                path_length = self.reconstruct_path_ucs(visited, start, goal)
                return path_length, states_visited, max_heap_size

            for neighbor in self.get_neighbors(current):
                new_cost = cost + 1
                if neighbor not in visited or new_cost < visited[neighbor][1]:
                    visited[neighbor] = (current, new_cost)
                    heapq.heappush(heap, (new_cost, neighbor))
                    states_visited += 1

        return -1, states_visited, max_heap_size

    def solve_bidirectional(self, board):
        """Bidirectional Search"""
        start = tuple(board)
        goal = self.goal

        if start == goal:
            return 0, 2, 0

        forward_queue = deque([start])
        backward_queue = deque([goal])
        forward_visited = {start: None}
        backward_visited = {goal: None}
        states_visited = 2
        max_space = 2

        while forward_queue or backward_queue:
            current_space = len(forward_visited) + len(backward_visited)
            max_space = max(max_space, current_space)

            if forward_queue:
                current = forward_queue.popleft()
                if current in backward_visited:
                    # Found intersection - calculate path length
                    forward_path = 0
                    temp = current
                    while temp != start and temp is not None:
                        temp = forward_visited[temp]
                        forward_path += 1

                    backward_path = 0
                    temp = current
                    while temp != goal and temp is not None:
                        temp = backward_visited[temp]
                        backward_path += 1

                    return forward_path + backward_path, states_visited, max_space

                for neighbor in self.get_neighbors(current):
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = current
                        forward_queue.append(neighbor)
                        states_visited += 1

            if backward_queue:
                current = backward_queue.popleft()
                if current in forward_visited:
                    # Found intersection
                    forward_path = 0
                    temp = current
                    while temp != start and temp is not None:
                        temp = forward_visited[temp]
                        forward_path += 1

                    backward_path = 0
                    temp = current
                    while temp != goal and temp is not None:
                        temp = backward_visited[temp]
                        backward_path += 1

                    return forward_path + backward_path, states_visited, max_space

                for neighbor in self.get_neighbors(current):
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = current
                        backward_queue.append(neighbor)
                        states_visited += 1

        return -1, states_visited, max_space

    def solve_greedy(self, board):
        """Greedy Best-First Search"""
        start = tuple(board)
        goal = self.goal

        if start == goal:
            return 0, 1, 0

        heap = [(self.manhattan_distance(start), start)]
        visited = {start: None}
        states_visited = 1
        max_heap_size = 1

        while heap:
            max_heap_size = max(max_heap_size, len(heap))
            current = heapq.heappop(heap)[1]

            if current == goal:
                path_length = self.reconstruct_path(visited, start, goal)
                return path_length, states_visited, max_heap_size

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    heapq.heappush(heap, (self.manhattan_distance(neighbor), neighbor))
                    states_visited += 1

        return -1, states_visited, max_heap_size

    def solve_astar(self, board):
        """A* Search"""
        start = tuple(board)
        goal = self.goal

        if start == goal:
            return 0, 1, 0

        heap = [(self.manhattan_distance(start), 0, start)]
        visited = {start: (None, 0)}
        states_visited = 1
        max_heap_size = 1

        while heap:
            max_heap_size = max(max_heap_size, len(heap))
            f_cost, g_cost, current = heapq.heappop(heap)

            if current == goal:
                path_length = self.reconstruct_path_ucs(visited, start, goal)
                return path_length, states_visited, max_heap_size

            for neighbor in self.get_neighbors(current):
                new_g_cost = g_cost + 1
                new_f_cost = new_g_cost + self.manhattan_distance(neighbor)

                if neighbor not in visited or new_g_cost < visited[neighbor][1]:
                    visited[neighbor] = (current, new_g_cost)
                    heapq.heappush(heap, (new_f_cost, new_g_cost, neighbor))
                    states_visited += 1

        return -1, states_visited, max_heap_size

    def solve_ida_star(self, board):
        """IDA* Search"""
        start = tuple(board)
        goal = self.goal

        if start == goal:
            return 0, 1, 0

        threshold = self.manhattan_distance(start)
        total_states = 0
        max_space = 0

        while threshold < 100:
            result, states, space = self.ida_star_search(start, goal, 0, threshold, [start])
            total_states += states
            max_space = max(max_space, space)
            if result != -1:
                return result, total_states, max_space
            if result == -2:  # No solution found
                break
            threshold += 1

        return -1, total_states, max_space

    def ida_star_search(self, current, goal, g_cost, threshold, path):
        """IDA* recursive search helper"""
        f_cost = g_cost + self.manhattan_distance(current)
        if f_cost > threshold:
            return -1, 1, len(path)
        if current == goal:
            return len(path) - 1, 1, len(path)

        states_visited = 1
        max_space = len(path)

        for neighbor in self.get_neighbors(current):
            if neighbor not in path:
                result, states, space = self.ida_star_search(neighbor, goal, g_cost + 1, threshold, path + [neighbor])
                states_visited += states
                max_space = max(max_space, space)
                if result != -1:
                    return result, states_visited, max_space

        return -1, states_visited, max_space

    def solve_puzzle(self, board, algorithm):
        """Solve puzzle using specified algorithm"""
        start_time = time.time()

        try:
            if algorithm == "BFS":
                moves, states, space = self.solve_bfs(board)
            elif algorithm == "DFS":
                moves, states, space = self.solve_dfs(board)
            elif algorithm == "IDS":
                moves, states, space = self.solve_ids(board)
            elif algorithm == "UCS":
                moves, states, space = self.solve_ucs(board)
            elif algorithm == "BiDir":
                moves, states, space = self.solve_bidirectional(board)
            elif algorithm == "Greedy":
                moves, states, space = self.solve_greedy(board)
            elif algorithm == "A*":
                moves, states, space = self.solve_astar(board)
            elif algorithm == "IDA*":
                moves, states, space = self.solve_ida_star(board)
            else:
                return None

            solve_time = time.time() - start_time

            return {
                'moves': moves if moves != -1 else None,
                'states_visited': states,
                'solution_time': solve_time,
                'space_complexity': space
            }
        except Exception as e:
            print(f"Error solving with {algorithm}: {e}")
            return None

def run_experiment():
    """Run the experiment with 10 different shuffles"""
    solver = PuzzleSolver()
    algorithms = solver.algorithms
    num_trials = 10

    # Initialize data storage
    results = {alg: {'moves': [], 'states': [], 'time': [], 'space': []} for alg in algorithms}

    print("Running 8-Puzzle Performance Analysis...")
    print("=" * 50)

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        board = solver.generate_random_board()
        print(f"Board: {board}")

        for algorithm in algorithms:
            print(f"  Running {algorithm}...", end=" ")
            result = solver.solve_puzzle(board, algorithm)

            if result and result['moves'] is not None:
                results[algorithm]['moves'].append(result['moves'])
                results[algorithm]['states'].append(result['states_visited'])
                results[algorithm]['time'].append(result['solution_time'] * 1000)  # Convert to ms
                results[algorithm]['space'].append(result['space_complexity'])
                print(f"✓ ({result['moves']} moves, {result['states_visited']} states)")
            else:
                print("✗ (No solution found)")

    return results

def calculate_averages(results):
    """Calculate average values for each metric"""
    averages = {}

    for algorithm in results:
        averages[algorithm] = {}
        for metric in ['moves', 'states', 'time', 'space']:
            values = results[algorithm][metric]
            if values:
                averages[algorithm][metric] = sum(values) / len(values)
            else:
                averages[algorithm][metric] = 0

    return averages

def plot_results(averages):
    """Create and display bar charts for each metric"""
    algorithms = list(averages.keys())
    metrics = [
        ('moves', 'Average Number of Moves', 'Moves'),
        ('states', 'Average States Visited', 'States'),
        ('time', 'Average Solution Time', 'Time (ms)'),
        ('space', 'Average Space Complexity', 'Memory Units')
    ]

    for metric_key, title, ylabel in metrics:
        values = [averages[alg][metric_key] for alg in algorithms]

        plt.figure(figsize=(12, 8))
        bars = plt.bar(algorithms, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                                 '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'])

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Search Algorithms', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        # Show the plot and wait for user to close it
        plt.show()

def main():
    """Main function to run the complete analysis"""
    print("8-Puzzle Performance Analysis")
    print("=" * 40)
    print("This program will:")
    print("1. Generate 10 random solvable puzzles")
    print("2. Solve each puzzle with all 8 algorithms")
    print("3. Calculate averages for each metric")
    print("4. Display results in separate bar charts")
    print("\nStarting analysis...")

    # Run the experiment
    results = run_experiment()

    # Calculate averages
    averages = calculate_averages(results)

    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 60)

    for algorithm in averages:
        print(f"\n{algorithm}:")
        print(f"  Average Moves: {averages[algorithm]['moves']:.2f}")
        print(f"  Average States Visited: {averages[algorithm]['states']:.2f}")
        print(f"  Average Solution Time: {averages[algorithm]['time']:.2f} ms")
        print(f"  Average Space Complexity: {averages[algorithm]['space']:.2f}")

    print("\n" + "=" * 60)
    print("Displaying bar charts...")
    print("Close each chart window to see the next one.")

    # Plot results
    plot_results(averages)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()