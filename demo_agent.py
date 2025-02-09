# demo_agent.py
import pygame
import torch
import game
import rl_agent

def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((game.WINDOW_WIDTH, game.WINDOW_HEIGHT + 100))
    pygame.display.set_caption("Toon Blast - AI Agent Demonstration")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24, bold=True)

    # Initialize RL environment and agent
    env = rl_agent.ToonBlastEnv()
    agent = rl_agent.DQNAgent(env)
    agent.q_net.load_state_dict(torch.load("best_model.pth", map_location=agent.device))
    agent.q_net.eval()
    state = env.reset()
    game_instance = env.game  # Reference to the actual game instance

    # Demonstration variables
    running = True
    last_action = None
    total_reward = 0
    move_counter = 0

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Agent decision making (only when idle and game not over)
        if not game_instance.is_game_over() and game_instance.anim_state == "idle":
            current_state = env.get_state()
            action = agent.get_action(current_state, eval_mode=True)
            
            if action is not None:
                row = action // game.BOARD_COLS
                col = action % game.BOARD_COLS
                prev_score = game_instance.score
                
                # Process the agent's move
                game_instance.process_move(row, col)
                move_counter += 1
                
                # Calculate rewards
                reward = game_instance.score - prev_score
                total_reward += reward
                last_action = (row, col)

        # Update game animations
        if game_instance.anim_state != "idle":
            game_instance.update_animations()

        # Draw game state
        screen.fill(game.BACKGROUND_COLOR)
        
        # Determine hidden cells during falling animations
        hidden_cells = set()
        if game_instance.anim_state == "fall_anim":
            for anim in game_instance.animations:
                if anim["type"] == "fall":
                    final_row = int(anim["end_y"] // game.CELL_SIZE)
                    hidden_cells.add((final_row, anim["col"]))
        
        # Draw game board and animations
        game.draw_board(screen, game_instance.board, font, hidden_cells)
        for anim in game_instance.animations:
            game.draw_animation(screen, anim, font)

        # Draw UI elements
        score_text = font.render(f"Score: {game_instance.score}", True, (40, 40, 40))
        moves_text = font.render(f"Moves Left: {game_instance.moves_remaining}", True, (40, 40, 40))
        target_text = font.render(f"Target: {game_instance.target_score}", True, (40, 40, 40))
        reward_text = font.render(f"Total Reward: {total_reward}", True, (0, 100, 0))
        
        screen.blit(score_text, (20, game.WINDOW_HEIGHT + 10))
        screen.blit(moves_text, (20, game.WINDOW_HEIGHT + 40))
        screen.blit(target_text, (20, game.WINDOW_HEIGHT + 70))
        screen.blit(reward_text, (game.WINDOW_WIDTH - 250, game.WINDOW_HEIGHT + 10))

        # Display last action taken
        if last_action:
            action_text = font.render(
                f"Last Move: ({last_action[0]}, {last_action[1]})", 
                True, (0, 0, 150)
            )
            screen.blit(action_text, (game.WINDOW_WIDTH - 250, game.WINDOW_HEIGHT + 40))

        # Game over display
        if game_instance.is_game_over():
            over_text = font.render("GAME OVER", True, (200, 0, 0))
            screen.blit(over_text, (game.WINDOW_WIDTH//2 - 60, game.WINDOW_HEIGHT + 50))
            
            if game_instance.score >= game_instance.target_score:
                result_text = font.render("Target Achieved!", True, (0, 150, 0))
            else:
                result_text = font.render("Target Missed", True, (150, 0, 0))
            screen.blit(result_text, (game.WINDOW_WIDTH//2 - 80, game.WINDOW_HEIGHT + 80))

        pygame.display.flip()
        clock.tick(game.FPS)

    pygame.quit()

if __name__ == "__main__":
    main()