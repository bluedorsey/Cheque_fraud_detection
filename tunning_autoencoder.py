import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
# DATASET

class GenuineImageDataset(Dataset):
    def __init__(self, folder, img_size=128):
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        exts       = {'.jpg','.jpeg','.png','.tiff','.bmp'}
        self.files = []
        folder     = Path(folder)
        if folder.exists():
            for f in folder.rglob('*'):
                if f.suffix.lower() in exts and f.is_file():
                    self.files.append(str(f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img)
        except:
            return torch.zeros(3, 128, 128)


# MODEL ARCHITECTURE

class DocumentAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        compressed    = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed


# TRAINING
def train_autoencoder(genuine_folder, save_path, doc_type='cheque', epochs=30, img_size=128):
    dataset = GenuineImageDataset(genuine_folder, img_size)
    if len(dataset) == 0:
        print(f'❌ No images found in: {genuine_folder}')
        return None

    print(f'Training autoencoder for {doc_type.upper()}')
    print(f'  Images found : {len(dataset)}')
    print(f'  Epochs       : {epochs}')
    print(f'  Image size   : {img_size}×{img_size}\n')

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Training on  : {device}')
    if torch.cuda.is_available():
        print(f'  GPU          : {torch.cuda.get_device_name(0)}')
    print()

    model     = DocumentAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss          = criterion(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            saved = '← best saved'
        else:
            saved = ''

        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{epochs}  |  Loss: {avg_loss:.6f}  {saved}')

    print(f'\n✅ Training complete!')
    print(f'   Best loss : {best_loss:.6f}')
    print(f'   Saved to  : {save_path}')
    return save_path


if __name__ == '__main__':
    # Configuration
    GENUINE_DIR = "D:/extra/cheque data" # Set this to your local genuine documents folder
    DOC_TYPE    = "cheque"
    SAVE_PATH   = f"models/autoencoder_{DOC_TYPE}.pt"

    print("Options:")
    print("1. Train Autoencoder")
    
    choice = input("Enter choice (1): ")

    if choice == '1':
        train_autoencoder(GENUINE_DIR, SAVE_PATH, doc_type=DOC_TYPE, epochs=30)
