import torch
import torch.nn as nn

class GameAI(nn.Module):
    def __init__(self):
        super(GameAI, self).__init__()
        
        # Конволюционная часть для обработки скриншотов
        self.conv_layers = nn.Sequential(
            # Первый блок: 128x227x1 -> 64x113x32
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Второй блок: 64x113x32 -> 32x56x64
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Третий блок: 32x56x64 -> 16x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Четвертый блок: 16x28x128 -> 8x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Ветка WASD: 4 выхода (w, a, s, d)
        self.wasd_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        
        # Ветка действий: 7 выходов (e, r, shift, ctrl, space, mouse_left, mouse_right)
        self.action_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 7),
            nn.Sigmoid()
        )
        
        # Ветка мыши: 2 выхода (mouse_dx, mouse_dy)
        self.mouse_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Tanh()
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, 128, 227)
        features = self.conv_layers(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)  # (batch_size, 256)
        
        wasd_output = self.wasd_branch(features)
        action_output = self.action_branch(features)
        mouse_output = self.mouse_branch(features)
        
        return {
            'wasd': wasd_output,
            'actions': action_output,
            'mouse': mouse_output
        }

if __name__ == "__main__":
    # Тест модели
    model = GameAI()
    print("Модель создана успешно!")
    
    # Тестовый прогон
    dummy_input = torch.randn(2, 1, 128, 227)
    output = model(dummy_input)
    
    print(f"WASD output shape: {output['wasd'].shape}")
    print(f"Actions output shape: {output['actions'].shape}")
    print(f"Mouse output shape: {output['mouse'].shape}")