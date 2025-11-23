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
        ТОЛЬКО для WASD и Actions (у них Sigmoid активация)
        Mouse оставляем как есть (Tanh активация)
        """
        wasd_binary = (predictions['wasd'] > wasd_threshold).astype(int)
        actions_binary = (predictions['actions'] > action_threshold).astype(int)
        
        return {
            'wasd': wasd_binary,
            'actions': actions_binary,
            'mouse': predictions['mouse']  # Мышь остается непрерывной (не применяем порог!)
        }
    
    def get_actions(self, frame, wasd_threshold=0.5, action_threshold=0.5):
        """
        Упрощенный метод для получения действий из кадра
        Возвращает словарь с готовыми к использованию действиями
        
        wasd_threshold: порог для W,A,S,D (0-1)
        action_threshold: порог для E,R,Shift,Ctrl,Space,MouseLeft,MouseRight (0-1)
        """
        # Получаем сырые предсказания от нейросети
        raw_predictions = self.predict(frame)
        
        # Применяем пороги ТОЛЬКО к бинарным выходам (WASD и Actions)
        binary_predictions = self.apply_threshold(
            raw_predictions, 
            wasd_threshold, 
            action_threshold
        )
        
        # Создаем удобный словарь действий
        actions = {
            # === ДВИЖЕНИЕ (WASD) ===
            # Sigmoid выходы → применяем порог
            'w': bool(binary_predictions['wasd'][0]),
            'a': bool(binary_predictions['wasd'][1]),
            's': bool(binary_predictions['wasd'][2]),
            'd': bool(binary_predictions['wasd'][3]),
            
            # === ДЕЙСТВИЯ ===
            # Sigmoid выходы → применяем порог
            'e': bool(binary_predictions['actions'][0]),
            'r': bool(binary_predictions['actions'][1]),
            'shift': bool(binary_predictions['actions'][2]),
            'ctrl': bool(binary_predictions['actions'][3]),
            'space': bool(binary_predictions['actions'][4]),
            'mouse_left': bool(binary_predictions['actions'][5]),
            'mouse_right': bool(binary_predictions['actions'][6]),
            
            # === МЫШЬ ===
            # Tanh выходы → НЕ применяем порог, оставляем непрерывные значения
            'mouse_dx': float(raw_predictions['mouse'][0]),  # Берем сырое значение!
            'mouse_dy': float(raw_predictions['mouse'][1])   # Берем сырое значение!
        }
        
        # Дополнительно: можно посмотреть уверенность модели
        if False:  # Поставь True для отладки
            print(f"WASD уверенности: {raw_predictions['wasd']}")
            print(f"Actions уверенности: {raw_predictions['actions']}")
            print(f"Mouse значения: {raw_predictions['mouse']}")
        
        return actions

    def get_actions_with_confidence(self, frame, wasd_threshold=0.5, action_threshold=0.5):
        """
        Расширенная версия, которая возвращает также уверенность модели
        Полезно для отладки и тонкой настройки
        """
        raw_predictions = self.predict(frame)
        binary_predictions = self.apply_threshold(raw_predictions, wasd_threshold, action_threshold)
        
        actions = {
            'w': bool(binary_predictions['wasd'][0]),
            'a': bool(binary_predictions['wasd'][1]),
            's': bool(binary_predictions['wasd'][2]),
            'd': bool(binary_predictions['wasd'][3]),
            
            'e': bool(binary_predictions['actions'][0]),
            'r': bool(binary_predictions['actions'][1]),
            'shift': bool(binary_predictions['actions'][2]),
            'ctrl': bool(binary_predictions['actions'][3]),
            'space': bool(binary_predictions['actions'][4]),
            'mouse_left': bool(binary_predictions['actions'][5]),
            'mouse_right': bool(binary_predictions['actions'][6]),
            
            'mouse_dx': float(raw_predictions['mouse'][0]),
            'mouse_dy': float(raw_predictions['mouse'][1])
        }
        
        # Дополнительная информация об уверенности
        confidence = {
            'wasd_confidence': raw_predictions['wasd'].tolist(),
            'actions_confidence': raw_predictions['actions'].tolist(),
            'mouse_values': raw_predictions['mouse'].tolist()
        }
        
        return actions, confidence

# Пример использования с отладкой
def demo():
    # Создаем инференс движок
    ai = GameAIInference()
    
    # Создаем тестовое черно-белое изображение
    test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    test_pil = Image.fromarray(test_frame)
    
    # Получаем предсказания с уверенностью
    actions, confidence = ai.get_actions_with_confidence(test_pil)
    
    print("Предсказанные действия:")
    for key, value in actions.items():
        if isinstance(value, bool):
            print(f"  {key}: {'✓' if value else '✗'}")
        else:
            print(f"  {key}: {value:.3f}")
    
    print("\nУверенность модели:")
    print(f"WASD: {confidence['wasd_confidence']}")
    print(f"Actions: {confidence['actions_confidence']}")
    print(f"Mouse: {confidence['mouse_values']}")

if __name__ == "__main__":
    demo()