# game.py
import pygame
import random
import math

# -------------------------------------------------
# Game Settings, Colors, and Constants
# -------------------------------------------------
BOARD_ROWS = 9
BOARD_COLS = 9
CELL_SIZE = 60
WINDOW_WIDTH = BOARD_COLS * CELL_SIZE
WINDOW_HEIGHT = BOARD_ROWS * CELL_SIZE
FPS = 30
COMBO_THRESHOLD = 10  # If a removal hits this many blocks, grant a combo extra.

# Colors (RGB)
COLORS = {
    "red":    (255, 0, 0),
    "green":  (0, 200, 0),
    "blue":   (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (160, 32, 240)
}
COLOR_LIST = list(COLORS.values())
BACKGROUND_COLOR = (250, 248, 239)

# Animation settings (in frames)
REMOVAL_ANIMATION_FRAMES = 15
FALL_ANIMATION_FRAMES = 15
MAX_EXPLOSION_RADIUS = CELL_SIZE // 2

# -------------------------------------------------
# Block and Block Creation Functions
# -------------------------------------------------
class Block:
    def __init__(self, block_type, color=None, health=0):
        """
        block_type: "normal", "bomb", "firework_v", "firework_h",
                    "balloon", "disco", "rocket", "box"
        """
        self.type = block_type
        self.color = color
        self.health = health

def new_block():
    r_val = random.random()
    if r_val < 0.03:
        return Block("bomb", color=random.choice(COLOR_LIST))
    elif r_val < 0.06:
        return Block("firework_v", color=random.choice(COLOR_LIST))
    elif r_val < 0.09:
        return Block("firework_h", color=random.choice(COLOR_LIST))
    elif r_val < 0.12:
        return Block("balloon", color=random.choice(COLOR_LIST))
    elif r_val < 0.15:
        return Block("disco", color=random.choice(COLOR_LIST))
    elif r_val < 0.25:
        return Block("box", color=(150, 150, 150), health=2)
    else:
        return Block("normal", color=random.choice(COLOR_LIST))

def spawn_extra():
    extra_types = ["rocket", "bomb", "disco"]
    chosen = random.choice(extra_types)
    return Block(chosen, color=random.choice(COLOR_LIST))

# -------------------------------------------------
# Drawing Helper Functions
# -------------------------------------------------
def draw_grid(screen):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

def draw_block_at(screen, block, x, y, font):
    """Draw a block at a given pixel coordinate (x,y)."""
    rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    if block is None:
        pygame.draw.rect(screen, BACKGROUND_COLOR, rect)
        return
    if block.type == "box":
        pygame.draw.rect(screen, block.color, rect)
        text = font.render(str(block.health), True, (0, 0, 0))
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)
    elif block.type == "bomb":
        pygame.draw.rect(screen, block.color, rect)
        center = rect.center
        radius = CELL_SIZE // 4
        pygame.draw.circle(screen, (50, 50, 50), center, radius)
        fuse_start = (center[0], center[1] - radius)
        fuse_end = (center[0], center[1] - radius - 10)
        pygame.draw.line(screen, (255, 0, 0), fuse_start, fuse_end, 3)
    elif block.type == "firework_v":
        pygame.draw.rect(screen, block.color, rect)
        center = rect.center
        radius = CELL_SIZE // 4
        pygame.draw.circle(screen, (255, 215, 0), center, radius)
        pygame.draw.line(screen, (255, 0, 0), (center[0], center[1]-radius), (center[0], center[1]+radius), 2)
    elif block.type == "firework_h":
        pygame.draw.rect(screen, block.color, rect)
        center = rect.center
        radius = CELL_SIZE // 4
        pygame.draw.circle(screen, (255, 215, 0), center, radius)
        pygame.draw.line(screen, (255, 0, 0), (center[0]-radius, center[1]), (center[0]+radius, center[1]), 2)
    elif block.type == "balloon":
        pygame.draw.rect(screen, block.color, rect)
        center = rect.center
        radius = CELL_SIZE // 4
        pygame.draw.circle(screen, (255, 255, 255), center, radius, 2)
    elif block.type == "disco":
        pygame.draw.rect(screen, block.color, rect)
        center = rect.center
        radius = CELL_SIZE // 4
        pygame.draw.circle(screen, block.color, center, radius)
        pygame.draw.circle(screen, (0, 0, 0), center, radius, 2)
        for angle in (45, 135, 225, 315):
            rad = math.radians(angle)
            start = (int(center[0] + radius * 0.5 * math.cos(rad)),
                     int(center[1] + radius * 0.5 * math.sin(rad)))
            end = (int(center[0] + radius * math.cos(rad)),
                   int(center[1] + radius * math.sin(rad)))
            pygame.draw.line(screen, (255, 255, 255), start, end, 2)
    elif block.type == "rocket":
        rocket_rect = pygame.Rect(x + 10, y + 10, CELL_SIZE - 20, CELL_SIZE - 20)
        pygame.draw.rect(screen, block.color, rocket_rect)
        tip = [(x + CELL_SIZE // 2, y + 5),
               (x + CELL_SIZE // 2 - 10, y + 20),
               (x + CELL_SIZE // 2 + 10, y + 20)]
        pygame.draw.polygon(screen, (255, 255, 255), tip)
    else:  # normal block
        pygame.draw.rect(screen, block.color, rect)
    pygame.draw.rect(screen, (0, 0, 0), rect, 1)

def draw_board(screen, board, font, hidden_cells=None):
    """
    Draws the board. If hidden_cells is provided (a set of (row, col)
    tuples), those cells are skipped (to allow for fall animation overlays).
    """
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if hidden_cells is not None and (r, c) in hidden_cells:
                continue
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            draw_block_at(screen, board[r][c], x, y, font)
    draw_grid(screen)

def draw_animation(screen, anim, font):
    """Draw an individual animation (pop or fall)."""
    if anim["type"] == "pop":
        # A "pop" animation shrinks the block as it disappears.
        factor = 1 - anim["frame"] / anim["total_frames"]
        x = anim["col"] * CELL_SIZE
        y = anim["row"] * CELL_SIZE
        center = (x + CELL_SIZE / 2, y + CELL_SIZE / 2)
        size = CELL_SIZE * factor
        rect = pygame.Rect(0, 0, size, size)
        rect.center = center
        block = anim["block"]
        if block.type == "box":
            pygame.draw.rect(screen, block.color, rect)
            text = font.render(str(block.health), True, (0, 0, 0))
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
        else:
            pygame.draw.rect(screen, block.color, rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    elif anim["type"] == "fall":
        # Linear interpolation from start_y to end_y.
        progress = anim["frame"] / anim["total_frames"]
        y = anim["start_y"] + progress * (anim["end_y"] - anim["start_y"])
        x = anim["col"] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        block = anim["block"]
        draw_block_at(screen, block, rect.x, rect.y, font)

# -------------------------------------------------
# Game Logic: ToonBlastGame Class with Animations
# -------------------------------------------------
class ToonBlastGame:
    def __init__(self):
        self.board = self.init_board()
        self.score = 0
        self.moves_remaining = 20
        self.target_score = 1000

        # Animation state:
        self.anim_state = "idle"  # can be "idle", "pop_anim", "fall_anim"
        self.animations = []      # list of active animations
        self.pending_board = None  # used to store the board after gravity

    def init_board(self):
        board = []
        for r in range(BOARD_ROWS):
            row = []
            for c in range(BOARD_COLS):
                row.append(new_block())
            board.append(row)
        return board

    def reset(self):
        self.board = self.init_board()
        self.score = 0
        self.moves_remaining = 20
        self.anim_state = "idle"
        self.animations = []
        self.pending_board = None

    def get_connected_group(self, row, col, color):
        """Flood–fill all adjacent (non-box) blocks with the same color."""
        visited = set()
        stack = [(row, col)]
        group = []
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            block = self.board[r][c]
            if block is None or block.type == "box":
                continue
            if block.color != color:
                continue
            group.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))
        return group

    def remove_blocks(self, positions):
        for r, c in positions:
            self.board[r][c] = None

    def damage_adjacent_boxes(self, positions):
        """For every removed cell, damage any adjacent box.
           If a box’s health drops to 0 or below, it is removed."""
        extra_to_remove = set()
        for r, c in positions:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                    block = self.board[nr][nc]
                    if block is not None and block.type == "box":
                        block.health -= 1
                        if block.health <= 0:
                            extra_to_remove.add((nr, nc))
        # Remove any boxes that were destroyed
        self.remove_blocks(extra_to_remove)
        return extra_to_remove

    def trigger_special_effect(self, row, col, block):
        """Return a set of positions to be removed as a result of a special block effect."""
        effect_positions = set()
        if block.type == "bomb":
            # Remove a 3x3 area around the bomb.
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                        effect_positions.add((nr, nc))
        elif block.type == "firework_v":
            for r in range(BOARD_ROWS):
                effect_positions.add((r, col))
        elif block.type == "firework_h":
            for c in range(BOARD_COLS):
                effect_positions.add((row, c))
        elif block.type == "balloon":
            effect_positions.add((row, col))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                    effect_positions.add((nr, nc))
        elif block.type == "disco":
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    other = self.board[r][c]
                    if other is not None and other.color == block.color:
                        effect_positions.add((r, c))
        elif block.type == "rocket":
            for c in range(BOARD_COLS):
                effect_positions.add((row, c))
            for r in range(BOARD_ROWS):
                effect_positions.add((r, col))
        else:
            effect_positions.add((row, col))
        return effect_positions

    def process_move(self, row, col):
        """Handles a click move at board position (row, col).
           Only processes input if no animation is in progress."""
        if self.moves_remaining <= 0 or self.anim_state != "idle":
            return  # either no moves or an animation is active

        block = self.board[row][col]
        if block is None:
            return

        removal_set = set()
        chain_queue = []

        special_types = ["bomb", "firework_v", "firework_h", "balloon", "disco", "rocket"]

        if block.type in special_types:
            chain_queue.append((row, col, block))
        else:
            group = self.get_connected_group(row, col, block.color)
            if len(group) < 2:
                return  # need at least 2 connected blocks
            removal_set.update(group)

        for (r, c) in list(removal_set):
            b = self.board[r][c]
            if b is not None and b.type in special_types:
                chain_queue.append((r, c, b))

        while chain_queue:
            r, c, sp_block = chain_queue.pop(0)
            effect_positions = self.trigger_special_effect(r, c, sp_block)
            for pos in effect_positions:
                if pos not in removal_set:
                    removal_set.add(pos)
                    b = self.board[pos[0]][pos[1]]
                    if b is not None and b.type in special_types:
                        chain_queue.append((pos[0], pos[1], b))

        extra_box_removals = self.damage_adjacent_boxes(removal_set)
        removal_set = removal_set.union(extra_box_removals)

        if not removal_set:
            return

        num_removed = len(removal_set)
        self.score += num_removed * 10

        # Combo bonus: if removal is large, spawn an extra block at the click.
        if num_removed >= COMBO_THRESHOLD and self.board[row][col] is None:
            self.board[row][col] = spawn_extra()

        self.moves_remaining -= 1

        # Schedule pop animations for the blocks to be removed.
        self.schedule_pop_animations(removal_set)

    def schedule_pop_animations(self, removal_set):
        """For each block in removal_set, create a pop (shrink) animation
           and remove it from the board immediately."""
        self.animations = []
        for (r, c) in removal_set:
            block = self.board[r][c]
            if block is None:
                continue
            anim = {
                "type": "pop",
                "row": r,
                "col": c,
                "block": block,
                "frame": 0,
                "total_frames": REMOVAL_ANIMATION_FRAMES
            }
            self.animations.append(anim)
            self.board[r][c] = None
        self.anim_state = "pop_anim"

    def schedule_fall_animations(self):
        """
        After removals are done, compute falling animations for each column.
        Instead of updating the board immediately, we compute a new board (pending_board)
        and schedule falling animations for blocks that shift downward as well as for
        newly spawned blocks.
        """
        new_board = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        animations = []
        for c in range(BOARD_COLS):
            # Build the current column from top to bottom.
            old_column = [self.board[r][c] for r in range(BOARD_ROWS)]
            non_empty = [(r, block) for r, block in enumerate(old_column) if block is not None]
            num_missing = BOARD_ROWS - len(non_empty)
            # New blocks fill the top.
            for i in range(num_missing):
                new_block_obj = new_block()
                new_board[i][c] = new_block_obj
                anim = {
                    "type": "fall",
                    "col": c,
                    "block": new_block_obj,
                    "start_y": -((num_missing - i) * CELL_SIZE),  # start off-screen
                    "end_y": i * CELL_SIZE,
                    "frame": 0,
                    "total_frames": FALL_ANIMATION_FRAMES
                }
                animations.append(anim)
            # Existing blocks fall down.
            for i, (old_r, block) in enumerate(non_empty):
                new_r = num_missing + i
                new_board[new_r][c] = block
                start_y = old_r * CELL_SIZE
                end_y = new_r * CELL_SIZE
                if start_y != end_y:
                    anim = {
                        "type": "fall",
                        "col": c,
                        "block": block,
                        "start_y": start_y,
                        "end_y": end_y,
                        "frame": 0,
                        "total_frames": FALL_ANIMATION_FRAMES
                    }
                    animations.append(anim)
            # For any cells that didn’t move, new_board already holds the block.
        self.pending_board = new_board
        self.animations = animations
        self.anim_state = "fall_anim"

    def update_animations(self):
        """Call this each frame to update active animations.
           When pop animations finish, trigger falling animations.
           When falling animations finish, update the board state."""
        if self.anim_state == "pop_anim":
            finished = []
            for anim in self.animations:
                anim["frame"] += 1
                if anim["frame"] >= anim["total_frames"]:
                    finished.append(anim)
            for anim in finished:
                self.animations.remove(anim)
            if not self.animations:
                # All pop animations finished; now schedule falling animations.
                self.schedule_fall_animations()
        elif self.anim_state == "fall_anim":
            finished = []
            for anim in self.animations:
                anim["frame"] += 1
                if anim["frame"] >= anim["total_frames"]:
                    finished.append(anim)
            for anim in finished:
                self.animations.remove(anim)
            if not self.animations:
                # Falling animations are done; update board to the new state.
                self.board = self.pending_board
                self.pending_board = None
                self.anim_state = "idle"

    def is_game_over(self):
        """Game over if no moves remain or if the target score is reached."""
        return self.moves_remaining <= 0 or self.score >= self.target_score

# -------------------------------------------------
# Custom Level Design Function (Optional)
# -------------------------------------------------
def design_custom_level(game: ToonBlastGame):
    """
    Sets a fixed custom level for demonstration:
      - Rows 0-2: red normal blocks (with a bomb at row 1, col 4)
      - Rows 3-5: blue normal blocks (with a horizontal firework at row 4, col 4)
      - Rows 6-8: green normal blocks (with a disco block at row 7, col 2)
    """
    game.board = []
    for r in range(BOARD_ROWS):
        row_list = []
        for c in range(BOARD_COLS):
            if r < 3:
                color = (255, 0, 0)
            elif r < 6:
                color = (0, 0, 255)
            else:
                color = (0, 200, 0)
            if r == 1 and c == 4:
                block = Block("bomb", color)
            elif r == 4 and c == 4:
                block = Block("firework_h", color)
            elif r == 7 and c == 2:
                block = Block("disco", color)
            else:
                block = Block("normal", color)
            row_list.append(block)
        game.board.append(row_list)

# -------------------------------------------------
# Main Game Loop
# -------------------------------------------------
def main():
    pygame.init()
    # Extra vertical space for score/move display.
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT + 100))
    pygame.display.set_caption("Toon Blast Clone with Animations")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    game = ToonBlastGame()
    # Uncomment the following line to use the custom level design:
    # design_custom_level(game)

    running = True
    while running:
        # Process events (only when no animation is active)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 
                  and game.anim_state == "idle"):
                mouse_x, mouse_y = event.pos
                if mouse_y < WINDOW_HEIGHT and not game.is_game_over():
                    col = mouse_x // CELL_SIZE
                    row = mouse_y // CELL_SIZE
                    game.process_move(row, col)

        # Update animations if active.
        if game.anim_state != "idle":
            game.update_animations()

        screen.fill(BACKGROUND_COLOR)
        # When falling animations are active, hide the board cells that are animating.
        hidden_cells = set()
        if game.anim_state == "fall_anim":
            for anim in game.animations:
                if anim["type"] == "fall":
                    # The final cell for this falling block
                    final_row = int(anim["end_y"] // CELL_SIZE)
                    hidden_cells.add((final_row, anim["col"]))
        draw_board(screen, game.board, font, hidden_cells)

        # Draw active animations (both pop and fall).
        for anim in game.animations:
            draw_animation(screen, anim, font)

        # Draw UI: Score, Moves, Target.
        score_text = font.render(f"Score: {game.score}", True, (0, 0, 0))
        moves_text = font.render(f"Moves: {game.moves_remaining}", True, (0, 0, 0))
        target_text = font.render(f"Target: {game.target_score}", True, (0, 0, 0))
        screen.blit(score_text, (10, WINDOW_HEIGHT + 10))
        screen.blit(moves_text, (10, WINDOW_HEIGHT + 40))
        screen.blit(target_text, (10, WINDOW_HEIGHT + 70))
        if game.is_game_over():
            over_text = font.render("Game Over!", True, (255, 0, 0))
            screen.blit(over_text, (WINDOW_WIDTH // 2 - 50, WINDOW_HEIGHT + 40))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()
