import mss
import cv2
import numpy as np
from PIL import Image
import time
import os
import threading
from pynput import keyboard, mouse
from pynput.mouse import Button
from pynput.keyboard import Key, Listener
import torch
from use import GameAIInference

class GameBot:
    def __init__(self, model_path='./models/game_ai_epoch_20.pth'):
        self.ai = GameAIInference(model_path)
        self.running = False
        self.activated = False  # Активирован ли бот
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30  # 30 FPS для стабильности
        
        # Инициализация для захвата экрана
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Основной монитор
        
        # Инициализация контроллеров
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        
        # Текущее состояние клавиш (чтобы не нажимать повторно)
        self.current_keys_pressed = set()
        
        # НАСТРОЙКИ МЫШИ - УВЕЛИЧЕНЫ ДЛЯ МАЛЕНЬКИХ ЗНАЧЕНИЙ
        self.mouse_sensitivity = 100  # УВЕЛИЧЕНО для маленьких значений (-0.07 до 0.07)
        self.mouse_deadzone = 0.000001     # Меньшая мертвая зона для чувствительности
        self.smooth_mouse = True       # Сглаживание движений мыши
        self.last_mouse_dx = 0
        self.last_mouse_dy = 0
        self.smoothing_factor = 0.6    # Коэффициент сглаживания
        
        # Hotkey listener
        self.hotkey_listener = None
        
        print("GameBot инициализирован!")
        print("Управление:")
        print("  = (равно) - активировать/деактивировать бота")
        print("  - (минус) - экстренная остановка")

    def capture_frame(self):
        """Захватывает текущий кадр с экрана"""
        try:
            screenshot = self.sct.grab(self.monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
        except Exception as e:
            print(f"Ошибка захвата экрана: {e}")
            return None

    def process_frame_for_ai(self, frame):
        """Обрабатывает кадр для нейросети (как в data collector)"""
        frame = frame.resize((227, 128))
        frame = frame.convert('L')
        return frame

    def smooth_mouse_movement(self, dx, dy):
        """Сглаживает движение мыши"""
        smoothed_dx = self.last_mouse_dx * self.smoothing_factor + dx * (1 - self.smoothing_factor)
        smoothed_dy = self.last_mouse_dy * self.smoothing_factor + dy * (1 - self.smoothing_factor)
        
        self.last_mouse_dx = smoothed_dx
        self.last_mouse_dy = smoothed_dy
        
        return int(smoothed_dx), int(smoothed_dy)

    def execute_actions(self, actions):
        """Выполняет предсказанные действия"""
        try:
            # Обрабатываем движение (WASD)
            wasd_keys = ['w', 'a', 's', 'd']
            for i, key in enumerate(wasd_keys):
                if actions[key] and key not in self.current_keys_pressed:
                    self.keyboard_controller.press(key)
                    self.current_keys_pressed.add(key)
                elif not actions[key] and key in self.current_keys_pressed:
                    self.keyboard_controller.release(key)
                    self.current_keys_pressed.discard(key)

            # Обрабатываем действия (E, R, Shift, Ctrl, Space)
            action_keys = [
                ('e', 'e'),
                ('r', 'r'),
                ('shift', Key.shift),
                ('ctrl', Key.ctrl),
                ('space', Key.space)
            ]
            
            for action_name, key in action_keys:
                if actions[action_name] and key not in self.current_keys_pressed:
                    self.keyboard_controller.press(key)
                    self.current_keys_pressed.add(key)
                elif not actions[action_name] and key in self.current_keys_pressed:
                    self.keyboard_controller.release(key)
                    self.current_keys_pressed.discard(key)

            # Обрабатываем кнопки мыши
            if actions['mouse_left']:
                self.mouse_controller.press(Button.left)
            else:
                self.mouse_controller.release(Button.left)
                
            if actions['mouse_right']:
                self.mouse_controller.press(Button.right)
            else:
                self.mouse_controller.release(Button.right)

            # Обрабатываем движение мыши (ОСНОВНОЕ ИСПРАВЛЕНИЕ)
            mouse_dx = actions['mouse_dx']
            mouse_dy = actions['mouse_dy']
            
            print(f"Мышь: dx={mouse_dx:.4f}, dy={mouse_dy:.4f}")  # Отладочная информация
            
            # Применяем мертвую зону
            if abs(mouse_dx) < self.mouse_deadzone:
                mouse_dx = 0
            if abs(mouse_dy) < self.mouse_deadzone:
                mouse_dy = 0
                
            # ПРЕОБРАЗУЕМ В ПИКСЕЛИ С БОЛЬШИМ МНОЖИТЕЛЕМ
            dx = int(mouse_dx * self.mouse_sensitivity)
            dy = int((mouse_dy + 0.04) * self.mouse_sensitivity*0)
            
            # Сглаживаем движение мыши
            if self.smooth_mouse and (dx != 0 or dy != 0):
                dx, dy = self.smooth_mouse_movement(dx, dy)
            
            # Двигаем мышь только если есть значительное движение
            if dx != 0 or dy != 0:
                self.mouse_controller.move(dx, dy)
                print(f"Движение мыши: {dx}, {dy}")  # Отладочная информация

        except Exception as e:
            print(f"Ошибка выполнения действий: {e}")

    def release_all_keys(self):
        """Отпускает все нажатые клавиши (экстренная остановка)"""
        for key in list(self.current_keys_pressed):
            try:
                self.keyboard_controller.release(key)
            except:
                pass
        self.current_keys_pressed.clear()
        
        # Отпускаем кнопки мыши
        self.mouse_controller.release(Button.left)
        self.mouse_controller.release(Button.right)
        
        # Сбрасываем состояние сглаживания мыши
        self.last_mouse_dx = 0
        self.last_mouse_dy = 0

    def game_loop(self):
        """Основной игровой цикл"""
        frame_count = 0
        start_time = time.time()
        
        print("Игровой цикл запущен!")
        
        while self.running:
            try:
                # Проверяем активацию
                if not self.activated:
                    time.sleep(0.02)
                    continue
                
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                # Поддерживаем стабильный FPS
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                
                # Захватываем и обрабатываем кадр
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                processed_frame = self.process_frame_for_ai(frame)
                
                # Получаем предсказания от нейросети
                actions = self.ai.get_actions(processed_frame, threshold=0.5)
                
                # Выполняем действия
                self.execute_actions(actions)
                
                # Статистика FPS
                frame_count += 1
                self.last_frame_time = time.time()
                
                if frame_count % 50 == 0:
                    fps = frame_count / (time.time() - start_time)
                    print(f"FPS: {fps:.1f}, Кадров обработано: {frame_count}")
                    
            except Exception as e:
                print(f"Ошибка в игровом цикле: {e}")
                time.sleep(0.1)

    def on_hotkey(self, key):
        """Обработчик горячих клавиш"""
        try:
            if hasattr(key, 'char'):
                if key.char == '=':  # Активация/деактивация
                    self.activated = not self.activated
                    status = "АКТИВИРОВАН" if self.activated else "ДЕАКТИВИРОВАН"
                    print(f"\n=== Бот {status} ===")
                    
                    if not self.activated:
                        self.release_all_keys()
                        
                elif key.char == '-':  # Экстренная остановка
                    print("\n=== ЭКСТРЕННАЯ ОСТАНОВКА ===")
                    self.activated = False
                    self.running = False
                    self.release_all_keys()
                    
        except AttributeError:
            pass

    def start_hotkey_listener(self):
        """Запускает слушатель горячих клавиш"""
        def on_press(key):
            self.on_hotkey(key)
        
        self.hotkey_listener = Listener(on_press=on_press)
        self.hotkey_listener.start()

    def start(self):
        """Запускает бота"""
        self.running = True
        self.start_hotkey_listener()
        
        print("=" * 50)
        print("CSGO AI Bot запущен!")
        print("Горячие клавиши:")
        print("  = (равно) - старт/стоп бота")
        print("  - (минус) - экстренная остановка")
        print("Настройки мыши:")
        print(f"  Чувствительность: {self.mouse_sensitivity}")
        print(f"  Мертвая зона: {self.mouse_deadzone}")
        print(f"  Сглаживание: {'ВКЛ' if self.smooth_mouse else 'ВЫКЛ'}")
        print("=" * 50)
        
        try:
            self.game_loop()
        except KeyboardInterrupt:
            print("\nОстановка по Ctrl+C...")
        except Exception as e:
            print(f"Критическая ошибка: {e}")
        finally:
            self.stop()

    def stop(self):
        """Останавливает бота"""
        print("Остановка бота...")
        self.running = False
        self.activated = False
        self.release_all_keys()
        
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        
        self.sct.close()
        print("Бот остановлен!")

def main():
    # Проверяем наличие модели
    model_path = './models/game_ai_epoch_20.pth'
    if not os.path.exists(model_path):
        print(f"Ошибка: Модель {model_path} не найдена!")
        print("Сначала обучите модель с помощью train.py")
        return
    
    # Создаем и запускаем бота
    bot = GameBot(model_path)
    bot.start()

if __name__ == "__main__":
    main()