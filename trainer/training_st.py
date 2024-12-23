import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model.video_transformer import *
from data_loader.ucf_cap_dataset import *

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

csv_file = '/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/dataset.csv'
dataset = UCF101Dataset(csv_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SpaceTimeTransformer(
    img_size=224,         # Resize frames to 224x224
    num_frames=8,         # Use 8 frames per video
    in_chans=3,           # RGB channels
    num_classes=101,      # UCF101 has 101 classes
    embed_dim=768,
    depth=12,
    num_heads=12,
    attention_style='frozen-in-time'
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        #inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")
