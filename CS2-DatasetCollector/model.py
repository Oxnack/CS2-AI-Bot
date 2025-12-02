import torch
import torch.nn as nn

class GameAI(nn.Module):
    def __init__(self, mouse_scale=50.0):
        super(GameAI, self).__init__()
        
        mouse_scale = 3.0
        self.mouse_scale = mouse_scale
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.wasd_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        
        # e, r, shift, ctrl, space, mouse_left, mouse_right
        self.action_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 7),
            nn.Sigmoid()
        )
        
        self.mouse_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
            nn.Tanh()
        )
    
    def forward(self, x, training=True):
        features = self.conv_layers(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        wasd_output = self.wasd_branch(features)
        action_output = self.action_branch(features)
        mouse_output = self.mouse_branch(features)
        
        if not training:
            mouse_output = mouse_output * self.mouse_scale
            mouse_output = torch.round(mouse_output)  
        else:
            mouse_output = mouse_output
        
        return {
            'wasd': wasd_output,
            'actions': action_output,
            'mouse': mouse_output
        }

    def set_mouse_scale(self, scale):
        self.mouse_scale = scale

if __name__ == "__main__":
    model = GameAI()
    print("Модель создана успешно!")
    
    dummy_input = torch.randn(2, 1, 128, 227)
    output = model(dummy_input)
    
    print(f"WASD output shape: {output['wasd'].shape}")
    print(f"Actions output shape: {output['actions'].shape}")
    print(f"Mouse output shape: {output['mouse'].shape}")