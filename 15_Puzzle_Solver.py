import pyxel
import random
import time
import heapq
from collections import deque

class App:
    def __init__(self):
        # Dimensions for 15-puzzle (4x4)
        self.N = 4
        self.goal = tuple(range(1, 16)) + (0,)  # (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0)
        self.base_size = 120  # Same size but smaller tiles
        self.tile_size = self.base_size // self.N  # 30px per tile
        self.left_panel_width = 80
        self.right_panel_width = 60
        self.screen_width = self.base_size + self.left_panel_width + self.right_panel_width
        self.screen_height = 132

        pyxel.init(self.screen_width, self.screen_height, "15 Puzzle Solver", display_scale=4)
        pyxel.mouse(True)
        self.board = list(range(16))
        self.moves = 0
        self.shuffle_board()
        
        # Solution animation variables
        self.solving = False
        self.solution_path = []
        self.current_step = 0
        self.animation_timer = 0
        self.animation_delay = 12  # Slightly slower for 15-puzzle
        self.states_visited = 0
        self.solution_time = 0
        self.solution_length = 0
        self.current_algorithm = ""
        
        # Button definitions
        self.buttons = [
            {"name": "BFS", "y": 10},
            {"name": "DFS", "y": 25},
            {"name": "IDS", "y": 40},
            {"name": "UCS", "y": 55},
            {"name": "BiDir", "y": 70},
            {"name": "Greedy", "y": 85},
            {"name": "A*", "y": 100},
            {"name": "IDA*", "y": 115}
        ]
        
        pyxel.run(self.update, self.draw)

    def shuffle_board(self):
        # Generate a simpler solvable puzzle for testing
        attempts = 0
        while attempts < 100:
            # Start from solved state and make random moves
            self.board = list(range(1, 16)) + [0]
            for _ in range(50):  # Make 50 random moves
                empty_idx = self.board.index(0)
                neighbors = self.get_empty_neighbors_from_board()
                if neighbors:
                    swap_idx = random.choice(neighbors)
                    self.board[empty_idx], self.board[swap_idx] = self.board[swap_idx], self.board[empty_idx]
            
            if not self.is_solved():
                break
            attempts += 1
        
        # Fallback: create a known solvable configuration
        if self.is_solved():
            self.board = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0]
        
        self.moves = 0
        self.solving = False
        self.solution_path = []
        self.states_visited = 0
        self.solution_time = 0
        self.solution_length = 0
        self.current_algorithm = ""

    def get_empty_neighbors_from_board(self):
        empty_index = self.board.index(0)
        row, col = empty_index // self.N, empty_index % self.N
        neighbors = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= row+dr < self.N and 0 <= col+dc < self.N:
                neighbors.append((row+dr)*self.N + (col+dc))
        return neighbors

    def is_solvable(self):
        # For 15-puzzle, solvability is more complex
        inversions = 0
        for i in range(len(self.board)):
            for j in range(i+1, len(self.board)):
                if self.board[i] and self.board[j] and self.board[i] > self.board[j]:
                    inversions += 1
        
        # Find row of empty space from bottom
        empty_row = self.N - (self.board.index(0) // self.N)
        
        # If grid width is odd, return true if number of inversions is even
        # If grid width is even, return true if:
        # - blank is on even row counting from bottom and inversions is odd
        # - blank is on odd row counting from bottom and inversions is even
        if self.N % 2 == 1:
            return inversions % 2 == 0
        else:
            if empty_row % 2 == 0:
                return inversions % 2 == 1
            else:
                return inversions % 2 == 0

    def is_solved(self):
        return tuple(self.board) == self.goal

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

    # BFS Implementation (with reasonable limits)
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
            if time.time() - start_time > 20.0:  # 5 second timeout
                break
                
            current = queue.popleft()
            
            if current == goal:
                path = self.reconstruct_path(visited, start, goal)
                return path, states_visited, time.time() - start_time
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
                    states_visited += 1
                    
                    if states_visited > 10000000:  # Memory limit
                        return [], states_visited, time.time() - start_time
        
        return [], states_visited, time.time() - start_time

    # DFS Implementation (with depth limit)
    def solve_dfs(self):
        start = tuple(self.board)
        goal = self.goal
        
        if start == goal:
            return [], 1, 0

        start_time = time.time()
        max_depth = 200
        states_visited = 0
        
        def dfs_recursive(current, depth, visited_path):
            nonlocal states_visited
            states_visited += 1
            
            if time.time() - start_time > 20.0:
                return None
            if states_visited > 1000000:
                return None
            if current == goal:
                return visited_path
            if depth >= max_depth:
                return None
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited_path:
                    result = dfs_recursive(neighbor, depth + 1, visited_path + [neighbor])
                    if result is not None:
                        return result
            return None
        
        path = dfs_recursive(start, 0, [start])
        if path:
            moves = self.path_to_moves(path)
            return moves, states_visited, time.time() - start_time
        
        return [], states_visited, time.time() - start_time

    # Simplified Iterative Deepening Search
    def solve_ids(self):
        start = tuple(self.board)
        goal = self.goal
        
        if start == goal:
            return [], 1, 0

        start_time = time.time()
        total_states = 0
        
        for depth in range(1, 21):  # Limited depth
            if time.time() - start_time > 10.0:  # 10 second timeout
                break
                
            result, states = self.dls(start, goal, depth)
            total_states += states
            if result is not None:
                return result, total_states, time.time() - start_time
            if total_states > 2000000:  # Memory limit
                break
        
        return [], total_states, time.time() - start_time

    def dls(self, start, goal, max_depth):
        states_visited = 0
        
        def dls_recursive(current, depth, path):
            nonlocal states_visited
            states_visited += 1
            
            if states_visited > 5000:  # Limit per depth level
                return None
            if current == goal:
                return path
            if depth >= max_depth:
                return None
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in path[-2:]:  # Avoid immediate cycles
                    result = dls_recursive(neighbor, depth + 1, path + [neighbor])
                    if result is not None:
                        return result
            return None
        
        path = dls_recursive(start, 0, [start])
        if path:
            return self.path_to_moves(path), states_visited
        return None, states_visited

    # Simplified UCS
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
            if time.time() - start_time > 10.0:
                break
                
            cost, current = heapq.heappop(heap)
            
            if current == goal:
                path = self.reconstruct_path_ucs(visited, start, goal)
                return path, states_visited, time.time() - start_time
            
            if cost > 300:  # Depth limit
                continue
                
            for neighbor in self.get_neighbors(current):
                new_cost = cost + 1
                if neighbor not in visited or new_cost < visited[neighbor][1]:
                    visited[neighbor] = (current, new_cost)
                    heapq.heappush(heap, (new_cost, neighbor))
                    states_visited += 1
                    
                    if states_visited > 1500000:
                        return [], states_visited, time.time() - start_time
        
        return [], states_visited, time.time() - start_time

    # Simplified Bidirectional Search
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
            if time.time() - start_time > 10.0:
                break
                
            heuristic, current = heapq.heappop(heap)
            
            if current == goal:
                path = self.reconstruct_path(visited, start, goal)
                return path, states_visited, time.time() - start_time
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    heapq.heappush(heap, (self.manhattan_distance(neighbor), neighbor))
                    states_visited += 1
                    
                    if states_visited > 15000:
                        return [], states_visited, time.time() - start_time
        
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
            if time.time() - start_time > 15.0:  # Longer timeout for A*
                break
                
            f_cost, g_cost, current = heapq.heappop(heap)
            
            if current == goal:
                path = self.reconstruct_path_ucs(visited, start, goal)
                return path, states_visited, time.time() - start_time
            
            if g_cost > 50:  # Higher depth limit for A*
                continue
                
            for neighbor in self.get_neighbors(current):
                new_g_cost = g_cost + 1
                new_f_cost = new_g_cost + self.manhattan_distance(neighbor)
                
                if neighbor not in visited or new_g_cost < visited[neighbor][1]:
                    visited[neighbor] = (current, new_g_cost)
                    heapq.heappush(heap, (new_f_cost, new_g_cost, neighbor))
                    states_visited += 1
                    
                    if states_visited > 25000:  # Higher limit for A*
                        return [], states_visited, time.time() - start_time
        
        return [], states_visited, time.time() - start_time

    # Simplified IDA*
    def solve_ida_star(self):
        start = tuple(self.board)
        goal = self.goal
        
        if start == goal:
            return [], 1, 0

        start_time = time.time()
        threshold = self.manhattan_distance(start)
        total_states = 0
        
        for iteration in range(10):  # Limit iterations
            if time.time() - start_time > 20.0:  # 20 second timeout
                break
                
            result, states, next_threshold = self.ida_star_search(start, goal, 0, threshold, [start])
            total_states += states
            if result is not None:
                return result, total_states, time.time() - start_time
            if next_threshold == float('inf') or total_states > 50000:
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
            if neighbor not in path[-2:]:  # Avoid immediate cycles
                result, states, next_t = self.ida_star_search(neighbor, goal, g_cost + 1, threshold, path + [neighbor])
                states_visited += states
                if result is not None:
                    return result, states_visited, 0
                if next_t < min_threshold:
                    min_threshold = next_t
                    
                if states_visited > 5000:  # Limit per iteration
                    break
        
        return None, states_visited, min_threshold

    def reconstruct_path(self, visited, start, goal):
        path = []
        current = goal
        while current != start and current in visited:
            prev = visited[current]
            if prev is None:
                break
            # Find which tile moved by comparing states
            empty_idx_current = current.index(0)
            empty_idx_prev = prev.index(0)
            # The tile that moved is at the previous empty position
            path.append(empty_idx_prev)
            current = prev
        path.reverse()
        return path

    def reconstruct_path_ucs(self, visited, start, goal):
        path = []
        current = goal
        while current != start and current in visited:
            prev_info = visited[current]
            if prev_info is None or prev_info[0] is None:
                break
            prev = prev_info[0]
            empty_idx_current = current.index(0)
            empty_idx_prev = prev.index(0)
            path.append(empty_idx_prev)
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
            prev_state = state_path[i-1]
            curr_state = state_path[i]
            empty_idx_prev = prev_state.index(0)
            moves.append(empty_idx_prev)
        return moves

    def start_solving(self, algorithm):
        if not self.solving:
            self.current_algorithm = algorithm
            print(f"\nStarting {algorithm} search for 15-puzzle...")
            
            try:
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
                    print(f"--- {algorithm} Solution ---")
                    print(f"Solution found in {len(path)} moves")
                    print(f"States visited: {states}")
                    print(f"Solution time: {solve_time*1000:.2f} ms")
                else:
                    print(f"No solution found with {algorithm} within time/memory limits")

            except Exception as e:
                print(f"Error in {algorithm}: {e}")
                self.current_algorithm = ""

    def update_animation(self):
        if not self.solving or self.current_step >= len(self.solution_path):
            self.solving = False
            return

        self.animation_timer += 1
        if self.animation_timer >= self.animation_delay:
            self.animation_timer = 0
            tile_index = self.solution_path[self.current_step]
            if tile_index in self.get_empty_neighbors():
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
        
        # Draw puzzle tiles (4x4 grid)
        puzzle_x = self.left_panel_width
        for i, value in enumerate(self.board):
            x = puzzle_x + (i % self.N) * self.tile_size
            y = (i // self.N) * self.tile_size
            if value != 0:
                pyxel.rect(x, y, self.tile_size-1, self.tile_size-1, 6)
                # Adjust text positioning for smaller tiles and 2-digit numbers
                text_x = x + self.tile_size//2 - (4 if value >= 10 else 2)
                text_y = y + self.tile_size//2 - 2
                pyxel.text(text_x, text_y, str(value), 5)

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
            pyxel.text(right_panel_x + 5, 55, f"Opt:{self.solution_length}", 7)
            # Show states visited (truncated if large)
            states_text = f"States:{self.states_visited}" if self.states_visited < 10000 else f"States:{self.states_visited//1000}k"
            pyxel.text(right_panel_x + 5, 65, states_text, 7)

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