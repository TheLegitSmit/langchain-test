import os
import pygame
import sys
import math
from dotenv import load_dotenv
from langchain.llms import OpenAI

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# LangChain setup
llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)

# Game settings
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 60
PLAYER_SPEED = 3
ROT_SPEED = 0.05
NPC_COUNT = 5
NPC_RADIUS = 0.3
INTERACT_DIST = 1.2

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Room layout (simple square room)
ROOM_SIZE = 8  # 8x8 units

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('AI Befriending Room')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 32)

# Player state
player_x, player_y = 1.5, 1.5
player_angle = 0
score = 0

# NPCs: (x, y, is_befriended, conversation_history)
npcs = []
for i in range(NPC_COUNT):
    angle = i * (2 * math.pi / NPC_COUNT)
    x = ROOM_SIZE/2 + math.cos(angle) * (ROOM_SIZE/2 - 1)
    y = ROOM_SIZE/2 + math.sin(angle) * (ROOM_SIZE/2 - 1)
    npcs.append({'x': x, 'y': y, 'befriended': False, 'history': []})

# Conversation state
in_conversation = False
current_npc = None
user_input = ''
conversation_log = []
befriend_threshold = 2  # Number of positive exchanges to befriend
min_user_lines = 2
min_npc_lines = 2

def is_friendly(text):
    friendly_words = ["friend", "happy", "glad", "like you", "nice", "awesome", "great", "welcome"]
    return any(word in text.lower() for word in friendly_words)

# Helper functions
def draw_minimap():
    scale = 40
    pygame.draw.rect(screen, WHITE, (0, 0, ROOM_SIZE*scale, ROOM_SIZE*scale), 2)
    # Draw player
    px, py = int(player_x*scale), int(player_y*scale)
    pygame.draw.circle(screen, BLUE, (px, py), 6)
    # Draw NPCs
    for npc in npcs:
        if not npc['befriended']:
            nx, ny = int(npc['x']*scale), int(npc['y']*scale)
            pygame.draw.circle(screen, RED, (nx, ny), 6)

def draw_3d_view():
    # Stub: Just draw a rectangle for the wall and circles for NPCs in front
    screen.fill((30, 30, 30))
    # Draw NPCs as simple human figures if in front of player
    for npc in npcs:
        if npc['befriended']:
            continue
        dx, dy = npc['x'] - player_x, npc['y'] - player_y
        dist = math.hypot(dx, dy)
        angle_to_npc = math.atan2(dy, dx) - player_angle
        if -math.pi/4 < angle_to_npc < math.pi/4 and dist < 6:
            # Project to screen
            size = max(20, int(200 / (dist+0.1)))
            x = SCREEN_WIDTH//2 + int(math.tan(angle_to_npc) * 300)
            y = SCREEN_HEIGHT//2
            # Draw body
            body_height = int(size * 2)
            body_width = int(size * 0.6)
            head_radius = int(size * 0.5)
            # Body
            pygame.draw.rect(screen, (220, 180, 140), (x - body_width//2, y + head_radius, body_width, body_height))
            # Head
            pygame.draw.circle(screen, (255, 224, 189), (x, y), head_radius)
            # Eyes
            eye_y = y - int(head_radius * 0.2)
            eye_dx = int(head_radius * 0.4)
            pygame.draw.circle(screen, BLACK, (x - eye_dx, eye_y), max(2, head_radius//8))
            pygame.draw.circle(screen, BLACK, (x + eye_dx, eye_y), max(2, head_radius//8))
            # Mouth
            mouth_y = y + int(head_radius * 0.3)
            mouth_w = int(head_radius * 0.5)
            pygame.draw.arc(screen, BLACK, (x - mouth_w//2, mouth_y, mouth_w, max(2, head_radius//4)), math.pi, 2*math.pi, 2)
    # Draw score
    score_text = font.render(f'Score: {score}', True, YELLOW)
    screen.blit(score_text, (10, 10))
    draw_minimap()

def get_facing_npc():
    for idx, npc in enumerate(npcs):
        if npc['befriended']:
            continue
        dx, dy = npc['x'] - player_x, npc['y'] - player_y
        dist = math.hypot(dx, dy)
        angle_to_npc = math.atan2(dy, dx) - player_angle
        if dist < INTERACT_DIST and abs(angle_to_npc) < math.pi/6:
            return idx, npc
    return None, None

def flash_effect():
    for i in range(6):
        screen.fill(YELLOW if i%2==0 else WHITE)
        pygame.display.flip()
        pygame.time.delay(60)

def openai_conversation(npc, user_message):
    # Use LangChain to get a response
    npc['history'].append({'role': 'user', 'content': user_message})
    prompt = "You are a friendly character in a game. Try to befriend the player.\n"
    for msg in npc['history'][-4:]:
        prompt += ("Player: " if msg['role']=='user' else "NPC: ") + msg['content'] + "\n"
    response = llm(prompt)
    npc['history'].append({'role': 'npc', 'content': response})
    return response

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def main():
    global player_x, player_y, player_angle, in_conversation, current_npc, user_input, score
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if in_conversation:
                    if event.key == pygame.K_RETURN:
                        if user_input.strip():
                            npc = npcs[current_npc]
                            response = openai_conversation(npc, user_input)
                            conversation_log.append(('You', user_input))
                            conversation_log.append(('NPC', response))
                            user_input = ''
                            # Befriend logic: require at least 2 user and 2 npc lines, and friendliness
                            user_lines = [m for m in npc['history'] if m['role']=='user']
                            npc_lines = [m for m in npc['history'] if m['role']=='npc']
                            if len(user_lines) >= min_user_lines and len(npc_lines) >= min_npc_lines and is_friendly(npc_lines[-1]['content']):
                                npc['befriended'] = True
                                flash_effect()
                                in_conversation = False
                                score += 1
                                if score == NPC_COUNT:
                                    print('You befriended everyone!')
                        continue
                    elif event.key == pygame.K_BACKSPACE:
                        user_input = user_input[:-1]
                    elif event.key <= 255:
                        user_input += event.unicode
                else:
                    if event.key == pygame.K_LEFT:
                        player_angle -= ROT_SPEED
                    elif event.key == pygame.K_RIGHT:
                        player_angle += ROT_SPEED
                    elif event.key == pygame.K_UP:
                        player_x += math.cos(player_angle) * PLAYER_SPEED * 0.1
                        player_y += math.sin(player_angle) * PLAYER_SPEED * 0.1
                        # Stay in room
                        player_x = max(1, min(ROOM_SIZE-1, player_x))
                        player_y = max(1, min(ROOM_SIZE-1, player_y))
                    elif event.key == pygame.K_DOWN:
                        player_x -= math.cos(player_angle) * PLAYER_SPEED * 0.1
                        player_y -= math.sin(player_angle) * PLAYER_SPEED * 0.1
                        player_x = max(1, min(ROOM_SIZE-1, player_x))
                        player_y = max(1, min(ROOM_SIZE-1, player_y))
                    elif event.key == pygame.K_SPACE:
                        idx, npc = get_facing_npc()
                        if npc:
                            in_conversation = True
                            current_npc = idx
                            conversation_log.clear()
                            conversation_log.append(('NPC', 'Hi! Want to be friends?'))
                            npc['history'].append({'role': 'npc', 'content': 'Hi! Want to be friends?'})
        # Draw
        draw_3d_view()
        if in_conversation:
            pygame.draw.rect(screen, BLACK, (50, SCREEN_HEIGHT-180, SCREEN_WIDTH-100, 160))
            y = SCREEN_HEIGHT-170
            for who, msg in conversation_log[-5:]:
                wrapped_lines = wrap_text(f'{who}: {msg}', font, SCREEN_WIDTH-120)
                for line in wrapped_lines:
                    txt = font.render(line, True, WHITE)
                    screen.blit(txt, (60, y))
                    y += 26
            # Draw input
            input_lines = wrap_text('> ' + user_input, font, SCREEN_WIDTH-120)
            iy = SCREEN_HEIGHT-40
            for line in input_lines:
                input_txt = font.render(line, True, GREEN)
                screen.blit(input_txt, (60, iy))
                iy += 26
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
