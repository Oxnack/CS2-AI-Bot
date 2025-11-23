import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from model import GameAI

class GameAIInference:
    def __init__(self, model_path='./models/game_ai_final.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")
        
        self.model = GameAI().to(self.device)
        self.load_model(model_path)
        self.model.eval()  # Режим оценки
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена: {path}")
    
    def process_frame(self, frame):
        """
        Обрабатывает кадр для нейросети
        На вход: PIL Image или numpy array
        Возвращает: тензор для нейросети (1, 1, 128, 227)
        """
        if isinstance(frame, np.ndarray):
            # Если numpy array, конвертируем в PIL
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = Image.fromarray(frame)
            elif len(frame.shape) == 2:
                frame = Image.fromarray(frame)
        
        # Конвертируем в grayscale и ресайзим
        if isinstance(frame, Image.Image):
            frame = frame.convert('L')
            frame = frame.resize((227, 128))  # width, height
        else:
            raise ValueError("Неподдерживаемый формат frame")
        
        # Конвертируем в numpy и нормализуем
        frame_array = np.array(frame, dtype=np.float32) / 255.0
        
        # Добавляем batch и channel размерности
        frame_tensor = torch.tensor(frame_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return frame_tensor.to(self.device)
    
    def predict(self, frame):
        """
        Делает предсказание для одного кадра
        На вход: PIL Image или numpy array
        Возвращает: словарь с предсказаниями
        """
        with torch.no_grad():
            # Обрабатываем кадр
            processed_frame = self.process_frame(frame)
            
            # Предсказание
            outputs = self.model(processed_frame)
            
            # Конвертируем в numpy и убираем batch dimension
            wasd = outputs['wasd'].cpu().numpy()[0]
            actions = outputs['actions'].cpu().numpy()[0]
            mouse = outputs['mouse'].cpu().numpy()[0]
            
            return {
                'wasd': wasd,
                'actions': actions,
                'mouse': mouse
            }
    
    def apply_threshold(self, predictions, wasd_threshold=0.5, action_threshold=0.5):
        """
        Применяет пороги для бинарных решений
        """
        wasd_binary = (predictions['wasd'] > wasd_threshold).astype(int)
        actions_binary = (predictions['actions'] > action_threshold).astype(int)
        
        return {
            'wasd': wasd_binary,
            'actions': actions_binary,
            'mouse': predictions['mouse']  # Мышь остается непрерывной
        }
    
    def get_actions(self, frame, threshold=0.5):
        """
        Упрощенный метод для получения действий из кадра
        Возвращает словарь с готовыми к использованию действиями
        """
        raw_predictions = self.predict(frame)
        binary_predictions = self.apply_threshold(raw_predictions, threshold, threshold)
        
        # Создаем удобный словарь действий
        actions = {
            # Движение
            'w': bool(binary_predictions['wasd'][0]),
            'a': bool(binary_predictions['wasd'][1]),
            's': bool(binary_predictions['wasd'][2]),
            'd': bool(binary_predictions['wasd'][3]),
            
            # Действия
            'e': bool(binary_predictions['actions'][0]),
            'r': bool(binary_predictions['actions'][1]),
            'shift': bool(binary_predictions['actions'][2]),
            'ctrl': bool(binary_predictions['actions'][3]),
            'space': bool(binary_predictions['actions'][4]),
            'mouse_left': bool(binary_predictions['actions'][5]),
            'mouse_right': bool(binary_predictions['actions'][6]),
            
            # Мышь
            'mouse_dx': float(binary_predictions['mouse'][0]),
            'mouse_dy': float(binary_predictions['mouse'][1])
        }
        
        return actions

# Пример использования
def demo():
    # Создаем инференс движок
    ai = GameAIInference()
    
    # Создаем тестовое черно-белое изображение (замени на реальный скриншот)
    test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)  # Черно-белое
    test_pil = Image.fromarray(test_frame)
    
    # Получаем предсказания
    actions = ai.get_actions(test_pil)
    
    print("Предсказанные действия:")
    for key, value in actions.items():
        if isinstance(value, bool):
            print(f"  {key}: {'✓' if value else '✗'}")
        else:
            print(f"  {key}: {value:.3f}")

if __name__ == "__main__":
    demo()