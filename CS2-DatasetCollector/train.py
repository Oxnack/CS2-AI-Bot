import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import GameAI

class GameDataset(Dataset):
    def __init__(self, images_folder, csv_path):
        self.images_folder = images_folder
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        
        # Загружаем и обрабатываем изображение
        img_path = os.path.join(self.images_folder, filename)
        img = Image.open(img_path).convert('L')  # Grayscale
        img = img.resize((227, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_array).unsqueeze(0)  # (1, 128, 227)
        
        # Подготавливаем таргеты
        wasd_target = torch.tensor([
            float(row['w']),
            float(row['a']),
            float(row['s']),
            float(row['d'])
        ], dtype=torch.float32)
        
        actions_target = torch.tensor([
            float(row['e']),
            float(row['r']),
            float(row['shift']),
            float(row['ctrl']),
            float(row['space']),
            float(row['mouse_left']),
            float(row['mouse_right'])
        ], dtype=torch.float32)
        
        mouse_target = torch.tensor([
            float(row['mouse_dx']),
            float(row['mouse_dy'])
        ], dtype=torch.float32)
        
        return img_tensor, {
            'wasd': wasd_target,
            'actions': actions_target,
            'mouse': mouse_target
        }

class GameAITrainer:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")
        
        self.model = GameAI().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Функции потерь
        self.wasd_criterion = nn.BCELoss()
        self.action_criterion = nn.BCELoss()
        self.mouse_criterion = nn.MSELoss()
        
        # Оптимизатор
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Планировщик обучения
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        wasd_loss = 0
        action_loss = 0
        mouse_loss = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Переносим данные на устройство
            images = images.to(self.device)
            wasd_targets = targets['wasd'].to(self.device)
            action_targets = targets['actions'].to(self.device)
            mouse_targets = targets['mouse'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Вычисляем потери
            wasd_loss_batch = self.wasd_criterion(outputs['wasd'], wasd_targets)
            action_loss_batch = self.action_criterion(outputs['actions'], action_targets)
            mouse_loss_batch = self.mouse_criterion(outputs['mouse'], mouse_targets)
            
            total_loss_batch = wasd_loss_batch + action_loss_batch + mouse_loss_batch
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Аккумулируем статистику
            total_loss += total_loss_batch.item()
            wasd_loss += wasd_loss_batch.item()
            action_loss += action_loss_batch.item()
            mouse_loss += mouse_loss_batch.item()
            
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {total_loss_batch.item():.4f}')
        
        # Усредняем потери
        total_loss /= len(dataloader)
        wasd_loss /= len(dataloader)
        action_loss /= len(dataloader)
        mouse_loss /= len(dataloader)
        
        return total_loss, wasd_loss, action_loss, mouse_loss
    
    def train(self, train_loader, epochs=50):
        print("Начинаем обучение...")
        
        for epoch in range(epochs):
            total_loss, wasd_loss, action_loss, mouse_loss = self.train_epoch(train_loader)
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Total Loss: {total_loss:.4f}')
            print(f'  WASD Loss: {wasd_loss:.4f}')
            print(f'  Action Loss: {action_loss:.4f}')
            print(f'  Mouse Loss: {mouse_loss:.4f}')
            
            # Сохраняем модель каждые 10 эпох
            if (epoch + 1) % 10 == 0:
                self.save_model(f'game_ai_epoch_{epoch+1}.pth')
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Модель сохранена: {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Модель загружена: {path}")

def main():
    # Параметры
    images_folder = './dataset/images'
    csv_path = './dataset/dataset.csv'
    batch_size = 32
    epochs = 50
    
    # Создаем датасет и даталоадер
    dataset = GameDataset(images_folder, csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"Загружено {len(dataset)} примеров")
    
    # Создаем тренер и начинаем обучение
    trainer = GameAITrainer()
    trainer.train(dataloader, epochs)
    
    # Сохраняем финальную модель
    trainer.save_model('game_ai_final.pth')

if __name__ == "__main__":
    main()