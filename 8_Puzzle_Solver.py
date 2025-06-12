import pyxel
import random
import time
import heapq
from collections import deque

class App:
    def __init__(self):
        # Dimensions
        self.N = 3
        self.goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.base_size = 120
        self.tile_size = self.base_size // self.N
        self.left_panel_width = 80
        self.right_panel_width = 60
        self.screen_width = self.base_size + self.left_panel_width + self.right_panel_width
        self.screen_height = 132

        pyxel.init(self.screen_width, self.screen_height, "8 Puzzle Solver", display_scale=4)
        pyxel.mouse(True)
        self.board = list(range(9))
        self.moves = 0
        self.shuffle_board()

        # Solution animation variables
        self.solving = False
        self.solution_path = []
        self.current_step = 0
        self.animation_timer = 0
        self.animation_delay = 8  # frames between moves
        self.states_visited = 0
        self.solution_time = 0
        self.solution_length = 0
        self.current_algorithm = ""

        # Button definitions
        self.buttons = [
            {"name": "BFS", "y": 5},
            {"name": "DFS", "y": 20},
            {"name": "IDS", "y": 35},
            {"name": "UCS", "y": 50},
            {"name": "BiDir", "y": 65},
            {"name": "Greedy", "y": 80},
            {"name": "A*", "y": 95},
            {"name": "IDA*", "y": 110}
        ]

        pyxel.run(self.update, self.draw)

    def shuffle_board(self):
        while True:
            self.board = random.sample(range(9), 9)
            if self.is_solvable() and not self.is_solved():
                break
        self.moves = 0
        self.solving = False
        self.solution_path = []
        self.states_visited = 0
        self.solution_time = 0
        self.solution_length = 0
        self.current_algorithm = ""

    def is_solvable(self):
        inversions = 0
        for i in range(len(self.board)):
            for j in range(i+1, len(self.board)):
                if self.board[i] and self.board[j] and self.board[i] > self.board[j]:
                    inversions += 1
        return inversions % 2 == 0

    def is_solved(self):
        return self.board[:8] == list(range(1, 9)) and self.board[8] == 0

    def get_empty_neighbors(self):
        empty_index = self.board.index(0)
        row, col = empty_index // self.N, empty_index % self.N
        neighbors = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= row+dr < self.N and 0 <= col+dc < self.N:
                neighbors.append((row+dr)*self.N + (col+dc))
        return neighbors

    def swap_tile(self, index):
        empty_index = self.board.index(0)
        self.board[empty_index], self.board[index] = self.board[index], self.board[empty_index]
        self.moves += 1

    def get_neighbors(self, state):
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
        distance = 0
        for i, value in enumerate(state):
            if value != 0:
                target_row, target_col = (value - 1) // self.N, (value - 1) % self.N
                current_row, current_col = i // self.N, i % self.N
                distance += abs(target_row - current_row) + abs(target_col - current_col)
        return distance

    def misplaced_tiles(self, state):
        return sum(1 for i, value in enumerate(state) if value != 0 and value != i + 1)

    # BFS
    def solve_bfs(self):
        start = tuple(self.board)
        goal = self.goal

        if start == goal:
            return [], 1, 0

        start_time = time.time()
        queue = deque([start])
        visited = {start: None}
        states_visited = 1

        while queue:
            current = queue.popleft()

            if current == goal:
                return self.reconstruct_path(visited, start, goal), states_visited, time.time() - start_time

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
                    states_visited += 1

        return [], states_visited, time.time() - start_time

    # DFS
    def solve_dfs(self):
        start = tuple(self.board)
        goal = self.goal

        if start == goal:
            return [], 1, 0

        start_time = time.time()
        stack = [start]
        visited = {start: None}
        states_visited = 1

        while stack:
            current = stack.pop()

            if current == goal:
                return self.reconstruct_path(visited, start, goal), states_visited, time.time() - start_time

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    stack.append(neighbor)
                    states_visited += 1

        return [], states_visited, time.time() - start_time

    # Iterative Deepening Search
    def solve_ids(self):
        start = tuple(self.board)
        goal = self.goal

        if start == goal:
            return [], 1, 0

        start_time = time.time()
        total_states = 0

        for depth in range(50):  # Max depth limit
            result, states = self.dls(start, goal, depth)
            total_states += states
            if result is not None:
                return result, total_states, time.time() - start_time

        return [], total_states, time.time() - start_time

    def dls(self, start, goal, max_depth):
        stack = [(start, 0, [start])]
        states_visited = 0

        while stack:
            current, depth, path = stack.pop()
            states_visited += 1

            if current == goal:
                return self.path_to_moves(path), states_visited

            if depth < max_depth:
                for neighbor in self.get_neighbors(current):
                    if neighbor not in path:  # Avoid cycles
                        stack.append((neighbor, depth + 1, path + [neighbor]))

        return None, states_visited

    # Uniform Cost Search
    def solve_ucs(self):
        start = tuple(self.board)
        goal = self.goal

        if start == goal:
            return [], 1, 0

        start_time = time.time()
        heap = [(0, start)]
        visited = {start: (None, 0)}
        states_visited = 1

        while heap:
            cost, current = heapq.heappop(heap)

            if current == goal:
                return self.reconstruct_path_ucs(visited, start, goal), states_visited, time.time() - start_time

            for neighbor in self.get_neighbors(current):
                new_cost = cost + 1
                if neighbor not in visited or new_cost < visited[neighbor][1]:
                    visited[neighbor] = (current, new_cost)
                    heapq.heappush(heap, (new_cost, neighbor))
                    states_visited += 1

        return [], states_visited, time.time() - start_time

    # Bidirectional Search
    def solve_bidirectional(self):
        start = tuple(self.board)
        goal = self.goal

        if start == goal:
            return [], 1, 0

        start_time = time.time()
        forward_queue = deque([start])
        backward_queue = deque([goal])
        forward_visited = {start: None}
        backward_visited = {goal: None}
        states_visited = 2

        while forward_queue or backward_queue:
            # Alternate between forward and backward search
            if forward_queue:
                current = forward_queue.popleft()
                if current in backward_visited:
                    return self.reconstruct_bidirectional_path(forward_visited, backward_visited, start, goal, current), states_visited, time.time() - start_time

                for neighbor in self.get_neighbors(current):
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = current
                        forward_queue.append(neighbor)
                        states_visited += 1
                        # Check if we found a connection
                        if neighbor in backward_visited:
                            return self.reconstruct_bidirectional_path(forward_visited, backward_visited, start, goal, neighbor), states_visited, time.time() - start_time

            if backward_queue:
                current = backward_queue.popleft()
                if current in forward_visited:
                    return self.reconstruct_bidirectional_path(forward_visited, backward_visited, start, goal, current), states_visited, time.time() - start_time

                for neighbor in self.get_neighbors(current):
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = current
                        backward_queue.append(neighbor)
                        states_visited += 1
                        # Check if we found a connection
                        if neighbor in forward_visited:
                            return self.reconstruct_bidirectional_path(forward_visited, backward_visited, start, goal, neighbor), states_visited, time.time() - start_time

        return [], states_visited, time.time() - start_time

    # Greedy Best-First Search
    def solve_greedy(self):
        start = tuple(self.board)
        goal = self.goal

        if start == goal:
            return [], 1, 0

        start_time = time.time()
        heap = [(self.manhattan_distance(start), start)]
        visited = {start: None}
        states_visited = 1

        while heap:
            current = heapq.heappop(heap)[1]

            if current == goal:
                return self.reconstruct_path(visited, start, goal), states_visited, time.time() - start_time

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    heapq.heappush(heap, (self.manhattan_distance(neighbor), neighbor))
                    states_visited += 1

        return [], states_visited, time.time() - start_time

    # A* Search
    def solve_astar(self):
        start = tuple(self.board)
        goal = self.goal

        if start == goal:
            return [], 1, 0

        start_time = time.time()
        heap = [(self.manhattan_distance(start), 0, start)]
        visited = {start: (None, 0)}
        states_visited = 1

        while heap:
            f_cost, g_cost, current = heapq.heappop(heap)

            if current == goal:
                return self.reconstruct_path_ucs(visited, start, goal), states_visited, time.time() - start_time

            for neighbor in self.get_neighbors(current):
                new_g_cost = g_cost + 1
                new_f_cost = new_g_cost + self.manhattan_distance(neighbor)

                if neighbor not in visited or new_g_cost < visited[neighbor][1]:
                    visited[neighbor] = (current, new_g_cost)
                    heapq.heappush(heap, (new_f_cost, new_g_cost, neighbor))
                    states_visited += 1

        return [], states_visited, time.time() - start_time

    # IDA* Search
    def solve_ida_star(self):
        start = tuple(self.board)
        goal = self.goal

        if start == goal:
            return [], 1, 0

        start_time = time.time()
        threshold = self.manhattan_distance(start)
        total_states = 0

        while threshold < 100:  # Max threshold limit
            result, states, next_threshold = self.ida_star_search(start, goal, 0, threshold, [start])
            total_states += states
            if result is not None:
                return result, total_states, time.time() - start_time
            if next_threshold == float('inf'):
                break
            threshold = next_threshold

        return [], total_states, time.time() - start_time

    def ida_star_search(self, current, goal, g_cost, threshold, path):
        f_cost = g_cost + self.manhattan_distance(current)
        if f_cost > threshold:
            return None, 1, f_cost
        if current == goal:
            return self.path_to_moves(path), 1, 0

        min_threshold = float('inf')
        states_visited = 1

        for neighbor in self.get_neighbors(current):
            if neighbor not in path:
                result, states, next_t = self.ida_star_search(neighbor, goal, g_cost + 1, threshold, path + [neighbor])
                states_visited += states
                if result is not None:
                    return result, states_visited, 0
                if next_t < min_threshold:
                    min_threshold = next_t

        return None, states_visited, min_threshold

    def reconstruct_path(self, visited, start, goal):
        path = []
        current = goal
        while current != start:
            prev = visited[current]
            # Find which tile moved
            empty_idx = current.index(0)
            path.append(empty_idx)
            current = prev
        path.reverse()
        return path

    def reconstruct_path_ucs(self, visited, start, goal):
        path = []
        current = goal
        while current != start:
            prev = visited[current][0]
            empty_idx = current.index(0)
            path.append(empty_idx)
            current = prev
        path.reverse()
        return path

    def reconstruct_bidirectional_path(self, forward_visited, backward_visited, start, goal, meeting_point):
        # Build forward path from start to meeting point
        forward_path = []
        current = meeting_point
        while current != start and current is not None:
            prev = forward_visited[current]
            if prev is not None:
                # Find the tile that moved from prev to current
                prev_empty = prev.index(0)
                curr_empty = current.index(0)
                # The tile that moved is at the current empty position in the previous state
                forward_path.append(curr_empty)
            current = prev
        forward_path.reverse()

        # Build backward path from meeting point to goal
        backward_path = []
        current = meeting_point
        while current != goal and current is not None:
            next_state = backward_visited[current]
            if next_state is not None:
                # Find the tile that moved from current to next_state
                curr_empty = current.index(0)
                next_empty = next_state.index(0)
                # The tile that moved is at the next empty position in the current state
                backward_path.append(next_empty)
                current = next_state
            else:
                break

        return forward_path + backward_path

    def path_to_moves(self, state_path):
        moves = []
        for i in range(1, len(state_path)):
            empty_idx = state_path[i].index(0)
            moves.append(empty_idx)
        return moves

    def start_solving(self, algorithm):
        if not self.solving:
            self.current_algorithm = algorithm
            start_time = time.time()

            if algorithm == "BFS":
                path, states, solve_time = self.solve_bfs()
            elif algorithm == "DFS":
                path, states, solve_time = self.solve_dfs()
            elif algorithm == "IDS":
                path, states, solve_time = self.solve_ids()
            elif algorithm == "UCS":
                path, states, solve_time = self.solve_ucs()
            elif algorithm == "BiDir":
                path, states, solve_time = self.solve_bidirectional()
            elif algorithm == "Greedy":
                path, states, solve_time = self.solve_greedy()
            elif algorithm == "A*":
                path, states, solve_time = self.solve_astar()
            elif algorithm == "IDA*":
                path, states, solve_time = self.solve_ida_star()
            else:
                return

            self.solution_path = path
            self.states_visited = states
            self.solution_time = solve_time
            self.solution_length = len(path)

            if path:
                self.solving = True
                self.current_step = 0
                self.animation_timer = 0
                print(f"\n--- {algorithm} Solution ---")
                print(f"Solution found in {len(path)} moves")
                print(f"States visited: {states}")
                print(f"Solution time: {solve_time*1000:.2f} ms")
            else:
                print(f"No solution found with {algorithm}")

    def update_animation(self):
        if not self.solving or self.current_step >= len(self.solution_path):
            self.solving = False
            return

        self.animation_timer += 1
        if self.animation_timer >= self.animation_delay:
            self.animation_timer = 0
            tile_index = self.solution_path[self.current_step]
            self.swap_tile(tile_index)
            self.current_step += 1

    def update(self):
        if self.solving:
            self.update_animation()

        if pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
            # Puzzle area interaction
            puzzle_x = self.left_panel_width
            if puzzle_x <= pyxel.mouse_x < puzzle_x + self.base_size:
                x = (pyxel.mouse_x - puzzle_x) // self.tile_size
                y = pyxel.mouse_y // self.tile_size
                if 0 <= y < self.N:
                    index = y * self.N + x
                    if index in self.get_empty_neighbors():
                        self.swap_tile(index)

            # Algorithm buttons (left panel)
            for button in self.buttons:
                if (5 <= pyxel.mouse_x <= 70 and
                    button["y"] <= pyxel.mouse_y <= button["y"] + 12):
                    self.start_solving(button["name"])

            # Shuffle button (right panel)
            btn_x = self.screen_width - self.right_panel_width + 5
            btn_y = self.screen_height - 25
            if (btn_x <= pyxel.mouse_x <= btn_x + 50 and
                btn_y <= pyxel.mouse_y <= btn_y + 20):
                self.shuffle_board()

    def draw(self):
        pyxel.cls(1)

        # Draw puzzle tiles
        puzzle_x = self.left_panel_width
        for i, value in enumerate(self.board):
            x = puzzle_x + (i % self.N) * self.tile_size
            y = (i // self.N) * self.tile_size
            if value != 0:
                pyxel.rect(x, y, self.tile_size-1, self.tile_size-1, 6)
                pyxel.text(x + self.tile_size//2 - 2, y + self.tile_size//2 - 2, str(value), 5)

        # Left panel - Algorithm buttons
        pyxel.rect(0, 0, self.left_panel_width, self.screen_height, 12)
        for button in self.buttons:
            color = 8 if self.current_algorithm == button["name"] else 5
            pyxel.rect(5, button["y"], 65, 12, color)
            pyxel.text(8, button["y"] + 3, button["name"], 7)

        # Right panel
        right_panel_x = self.screen_width - self.right_panel_width
        pyxel.rect(right_panel_x, 0, self.right_panel_width, self.screen_height, 12)

        # Moves counter
        pyxel.text(right_panel_x + 5, 15, "MOVES:", 0)
        pyxel.text(right_panel_x + 5, 25, str(self.moves), 7)

        # Stats
        if self.solution_length > 0:
            pyxel.text(right_panel_x + 5, 45, "STATS:", 0)
            pyxel.text(right_panel_x + 5, 55, f"Move:{self.solution_length}", 7)
            pyxel.text(right_panel_x + 5, 65, f"States:{self.states_visited}", 7)

        # Shuffle button
        btn_x = right_panel_x + 5
        btn_y = self.screen_height - 25
        pyxel.rect(btn_x, btn_y, 50, 20, 5)
        pyxel.text(btn_x + 8, btn_y + 7, "SHUFFLE", 7)

        # Solving indicator
        if self.solving:
            pyxel.text(right_panel_x + 5, 85, "SOLVING", 7)
            pyxel.text(right_panel_x + 5, 95, self.current_algorithm, 10)

        # Victory message
        if self.is_solved() and not self.solving:
            overlay_x = puzzle_x + 10
            overlay_y = 50
            pyxel.rect(overlay_x, overlay_y, 100, 20, 0)
            pyxel.text(overlay_x + 10, overlay_y + 7, "SOLVED!", 10)

App()