import cv2
import random
import numpy as np
import mediapipe as mp
import pygame
import sys
import threading
import time

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
GRID_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * GRID_SIZE) // 2
GRID_OFFSET_Y = 50

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
PURPLE = (180, 0, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
SHAPES = [
    [[1],[1],[1],[1]],  # I
    [[1, 1], [1, 1]],  # O
    [[0, 1, 0], [1, 1, 1]],  # T
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]]  # L
]


SHAPE_COLORS = [CYAN, YELLOW, PURPLE, GREEN, RED, BLUE, ORANGE]


class Tetromino:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.shape_idx = random.randint(0, len(SHAPES) - 1)
        self.shape = SHAPES[self.shape_idx]
        self.color = SHAPE_COLORS[self.shape_idx]
        self.rotation = 0

    def rotate(self):
        rows = len(self.shape)
        cols = len(self.shape[0])
        rotated = [[0 for _ in range(rows)] for _ in range(cols)]
        for r in range(rows):
            for c in range(cols):
                rotated[c][rows - 1 - r] = self.shape[r][c]
        return rotated


class TetrisGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Тетрис с управлением жестами")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)
        self.reset_game()
        # Для управления жестами
        self.hand_control = {"left": False, "right": False, "down": False, "rotate": False, "drop": False}
        self.last_gesture_time = time.time()

    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.fall_speed = 0.5
        self.fall_time = 0

    def new_piece(self):
        return Tetromino(GRID_WIDTH // 2 - 1, 0)

    def valid_position(self, piece, x, y, shape=None):
        shape_to_check = shape if shape else piece.shape
        for r, row in enumerate(shape_to_check):
            for c, cell in enumerate(row):
                if cell:
                    pos_x, pos_y = x + c, y + r
                    if (pos_x < 0 or pos_x >= GRID_WIDTH or
                            pos_y >= GRID_HEIGHT or
                            (pos_y >= 0 and self.grid[pos_y][pos_x])):
                        return False
        return True

    def merge_piece(self):
        for r, row in enumerate(self.current_piece.shape):
            for c, cell in enumerate(row):
                if cell:
                    pos_y = self.current_piece.y + r
                    pos_x = self.current_piece.x + c
                    if 0 <= pos_y < GRID_HEIGHT and 0 <= pos_x < GRID_WIDTH:
                        self.grid[pos_y][pos_x] = self.current_piece.color

    def clear_lines(self):
        lines_to_clear = []
        for i, row in enumerate(self.grid):
            if all(cell != 0 for cell in row):
                lines_to_clear.append(i)
        for line in lines_to_clear:
            del self.grid[line]
            self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])

        if lines_to_clear:
            self.lines_cleared += len(lines_to_clear)
            self.score += [100, 300, 500, 800][min(len(lines_to_clear) - 1, 3)] * self.level
            self.level = self.lines_cleared // 10 + 1
            self.fall_speed = max(0.05, 0.5 - (self.level - 1) * 0.05)

    def move(self, dx, dy):
        if not self.game_over:
            if self.valid_position(self.current_piece, self.current_piece.x + dx, self.current_piece.y + dy):
                self.current_piece.x += dx
                self.current_piece.y += dy
                return True
        return False

    def rotate_piece(self):
        if not self.game_over:
            rotated_shape = self.current_piece.rotate()
            if self.valid_position(self.current_piece, self.current_piece.x, self.current_piece.y, rotated_shape):
                self.current_piece.shape = rotated_shape
                return True
        return False

    def drop_piece(self):
        if not self.game_over:
            while self.move(0, 1):
                pass
            self.lock_piece()

    def lock_piece(self):
        self.merge_piece()
        self.clear_lines()
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()
        if not self.valid_position(self.current_piece, self.current_piece.x, self.current_piece.y):
            self.game_over = True

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pygame.draw.rect(
                    self.screen,
                    GRAY,
                    (GRID_OFFSET_X + x * GRID_SIZE, GRID_OFFSET_Y + y * GRID_SIZE, GRID_SIZE, GRID_SIZE),
                    1
                )
                if self.grid[y][x]:
                    pygame.draw.rect(
                        self.screen,
                        self.grid[y][x],
                        (GRID_OFFSET_X + x * GRID_SIZE + 1, GRID_OFFSET_Y + y * GRID_SIZE + 1, GRID_SIZE - 2,
                         GRID_SIZE - 2)
                    )

    def draw_current_piece(self):
        if not self.game_over:
            for r, row in enumerate(self.current_piece.shape):
                for c, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(
                            self.screen,
                            self.current_piece.color,
                            (GRID_OFFSET_X + (self.current_piece.x + c) * GRID_SIZE + 1,
                             GRID_OFFSET_Y + (self.current_piece.y + r) * GRID_SIZE + 1,
                             GRID_SIZE - 2, GRID_SIZE - 2)
                        )

    def draw_next_piece(self):
        next_text = self.font.render("Следующая:", True, WHITE)
        self.screen.blit(next_text, (GRID_OFFSET_X + GRID_WIDTH * GRID_SIZE + 20, GRID_OFFSET_Y))
        for r, row in enumerate(self.next_piece.shape):
            for c, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        self.screen,
                        self.next_piece.color,
                        (GRID_OFFSET_X + GRID_WIDTH * GRID_SIZE + 50 + c * GRID_SIZE,
                         GRID_OFFSET_Y + 50 + r * GRID_SIZE,
                         GRID_SIZE - 2, GRID_SIZE - 2)
                    )

    def draw_info(self):
        score_text = self.font.render(f"Счет: {self.score}", True, WHITE)
        level_text = self.font.render(f"Уровень: {self.level}", True, WHITE)
        lines_text = self.font.render(f"Линии: {self.lines_cleared}", True, WHITE)
        self.screen.blit(score_text, (GRID_OFFSET_X + GRID_WIDTH * GRID_SIZE + 20, GRID_OFFSET_Y + 150))
        self.screen.blit(level_text, (GRID_OFFSET_X + GRID_WIDTH * GRID_SIZE + 20, GRID_OFFSET_Y + 200))
        self.screen.blit(lines_text, (GRID_OFFSET_X + GRID_WIDTH * GRID_SIZE + 20, GRID_OFFSET_Y + 250))

    def draw_game_over(self):
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            game_over_text = self.font.render("ИГРА ОКОНЧЕНА", True, RED)
            restart_text = self.font.render("Нажмите R для новой игры", True, WHITE)
            self.screen.blit(game_over_text,
                             (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

    def draw_title(self):
        title_font = pygame.font.SysFont(None, 60)
        title = title_font.render("ТЕТРИС", True, YELLOW)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 10))

    def process_hand_gestures(self):
        current_time = time.time()
        if current_time - self.last_gesture_time < 0.2:  # 200 мс
            return
        self.last_gesture_time = current_time
        if self.hand_control["left"]:
            self.move(-1, 0)
        elif self.hand_control["right"]:
            self.move(1, 0)
        elif self.hand_control["down"]:
            self.move(0, 1)
        elif self.hand_control["rotate"]:
            self.rotate_piece()
        elif self.hand_control["drop"]:
            self.drop_piece()

    def run(self):
        last_time = pygame.time.get_ticks()
        running = True
        while running:
            current_time = pygame.time.get_ticks()
            delta_time = (current_time - last_time) / 1000.0
            last_time = current_time
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    if not self.game_over:
                        if event.key == pygame.K_LEFT:
                            self.move(-1, 0)
                        elif event.key == pygame.K_RIGHT:
                            self.move(1, 0)
                        elif event.key == pygame.K_DOWN:
                            self.move(0, 1)
                        elif event.key == pygame.K_UP:
                            self.rotate_piece()
                        elif event.key == pygame.K_SPACE:
                            self.drop_piece()
            self.process_hand_gestures()
            if not self.game_over:
                self.fall_time += delta_time
                if self.fall_time >= self.fall_speed:
                    if not self.move(0, 1):
                        self.lock_piece()
                    self.fall_time = 0
            self.screen.fill(BLACK)
            self.draw_title()
            self.draw_grid()
            self.draw_current_piece()
            self.draw_next_piece()
            self.draw_info()
            self.draw_game_over()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()
def process_video_feed(game):
    camera = cv2.VideoCapture(0)
    camera.set(3, 1280)  # Ширина
    camera.set(4, 960)  # Высота
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mpDraw = mp.solutions.drawing_utils
    prev_x = None
    prev_y = None
    gesture_start_time = None

    while True:
        success, img = camera.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # Сброс жестов
        game.hand_control = {"left": False, "right": False, "down": False, "rotate": False, "drop": False}

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Рисуем landmarks
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                # Получаем координаты указательного пальца (id=8)
                h, w, c = img.shape
                index_finger_tip = handLms.landmark[8]
                cx, cy = int(index_finger_tip.x * w), int(
                    index_finger_tip.y * h)

                # Получаем координаты большого пальца (id=4) для определения жестов
                thumb_tip = handLms.landmark[4]
                thumb_cx, thumb_cy = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Получаем координаты среднего пальца (id=12)
                middle_tip = handLms.landmark[12]
                middle_cx, middle_cy = int(middle_tip.x * w), int(middle_tip.y * h)

                # Рисуем точки на пальцах
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (thumb_cx, thumb_cy), 10, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, (middle_cx, middle_cy), 10, (0, 255, 255), cv2.FILLED)

                distance_thumb_index = np.sqrt((thumb_cx - cx) ** 2 + (thumb_cy - cy) ** 2)
                distance_middle_index = np.sqrt((middle_cx - cx) ** 2 + (middle_cy - cy) ** 2)

                # Жест "дырочка" - поворот
                if distance_thumb_index < 40:
                    game.hand_control["rotate"] = True
                    cv2.putText(img, "rotate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                # Движение влево/вправо/вниз по позиции указательного пальца
                else:
                    if prev_x is not None and prev_y is not None:
                        dx = cx - prev_x
                        dy = cy - prev_y

                        threshold = 15

                        if abs(dx) > abs(dy) and abs(dx) > threshold:
                            if dx > 0:
                                game.hand_control["right"] = True
                                cv2.putText(img, "right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                game.hand_control["left"] = True
                                cv2.putText(img, "left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif abs(dy) > abs(dx) and abs(dy) > threshold:
                            if dy > 0:
                                game.hand_control["down"] = True
                                cv2.putText(img, "down", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                pass
                prev_x = cx
                prev_y = cy

        cv2.imshow("Tetris", img)

        # Выход по клавише ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    game = TetrisGame()
    video_thread = threading.Thread(target=process_video_feed, args=(game,))
    video_thread.daemon = True
    video_thread.start()

    game.run()