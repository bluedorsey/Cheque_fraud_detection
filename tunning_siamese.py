import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

# FIND IMAGES

def find_images(folder):
    exts   = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    folder = Path(folder)
    if not folder.exists():
        return []
    return [str(f) for f in folder.rglob('*')
            if f.is_file() and f.suffix.lower() in exts]


# SIGNATURE TRANSFORM
sig_transform = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# MODEL DEFINITION
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        backbone        = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc     = nn.Identity()

        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone   = backbone
        self.comparator = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        return self.backbone(x)

    def forward(self, img1, img2):
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        diff  = torch.abs(feat1 - feat2)
        return self.comparator(diff)


# DATASET

class SignatureDataset(Dataset):
    def __init__(self, sig_dir, transform, n_pairs=5000):
        self.transform = transform
        self.pairs     = []

        sig_dir = Path(sig_dir)
        if not sig_dir.exists():
            print(f'❌ Folder not found: {sig_dir}')
            return

        all_folders  = [f for f in sig_dir.iterdir() if f.is_dir()]
        genuine_dirs = sorted([f for f in all_folders if '_forg' not in f.name])
        forged_dirs  = sorted([f for f in all_folders if '_forg'     in f.name])

        print(f'People found    : {len(genuine_dirs)}')
        print(f'Genuine folders : {len(genuine_dirs)}')
        print(f'Forged  folders : {len(forged_dirs)}')

        people = {}
        for gdir in genuine_dirs:
            person_id = gdir.name
            forg_dir  = sig_dir / f'{person_id}_forg'
            g_images  = find_images(str(gdir))
            f_images  = find_images(str(forg_dir)) if forg_dir.exists() else []
            if len(g_images) >= 2:
                people[person_id] = {
                    'genuine': g_images,
                    'forged' : f_images,
                }

        print(f'Usable people   : {len(people)}')
        if len(people) == 0:
            print('❌ No people found — check SIGNATURES_DIR path')
            return

        person_ids = list(people.keys())
        for _ in range(n_pairs // 2):

            pid    = random.choice(person_ids)
            g_imgs = people[pid]['genuine']
            if len(g_imgs) >= 2:
                g1, g2 = random.sample(g_imgs, 2)
                self.pairs.append((g1, g2, 1))

            pid    = random.choice(person_ids)
            g_imgs = people[pid]['genuine']
            f_imgs = people[pid]['forged']
            if g_imgs and f_imgs:
                g = random.choice(g_imgs)
                f = random.choice(f_imgs)
                self.pairs.append((g, f, 0))

        random.shuffle(self.pairs)
        print(f'Total pairs     : {len(self.pairs)}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        try:
            img1 = self.transform(Image.open(path1).convert('RGB'))
            img2 = self.transform(Image.open(path2).convert('RGB'))
        except:
            img1 = torch.zeros(3, 224, 224)
            img2 = torch.zeros(3, 224, 224)
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# TRAINING

def train_siamese(sig_dir, save_path, epochs=20, n_pairs=5000):

    dataset = SignatureDataset(sig_dir, sig_transform, n_pairs=n_pairs)
    if len(dataset) == 0:
        print('❌No pairs loaded')
        return None

    val_size   = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Train pairs : {train_size}')
    print(f'Val pairs   : {val_size}')
    print(f'Device      : {device}\n')

    model     = SiameseNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    patience      = 0

    for epoch in range(1, epochs + 1):

        model.train()
        train_loss = 0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            preds  = model(img1, img2).squeeze()
            loss   = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss    = 0
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                preds    = model(img1, img2).squeeze()
                val_loss += criterion(preds, labels).item()
                correct   = ((preds > 0.5).float() == labels).sum().item()
                val_correct += correct
                val_total   += len(labels)

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        val_acc   = val_correct / val_total * 100

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            saved   = '← best saved'
            patience = 0
        else:
            saved    = ''
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{epochs}  |  '
                  f'Train: {avg_train:.4f}  '
                  f'Val: {avg_val:.4f}  '
                  f'Val Acc: {val_acc:.1f}%  {saved}')

        if patience >= 5:
            print(f'Early stopping at epoch {epoch} — val loss not improving')
            break

    print(f'\nTraining complete  |  Best val loss: {best_val_loss:.6f}')
    return model



# INFERENCE PIPELINE
def extract_signature_region(image_rgb, doc_type='cheque'):
    h, w = image_rgb.shape[:2]
    if doc_type == 'cheque':
        region = image_rgb[int(h*0.65):int(h*0.88), int(w*0.55):int(w*0.98)]
        if region.size == 0:
            return None
        return Image.fromarray(region)
    return None

def load_siamese(model_path):
    if not Path(model_path).exists():
        return None
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def compare_signatures(sig1_pil, sig2_pil, model_path):
    model = load_siamese(model_path)
    if model is None:
        print('Using untrained model — run train_siamese() for accuracy')
        model = SiameseNetwork()
        model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    t1 = sig_transform(sig1_pil).unsqueeze(0).to(device)
    t2 = sig_transform(sig2_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        similarity = model(t1, t2).item()

    flagged = similarity < 0.45

    return {
        'similarity'     : round(similarity, 4),
        'signature_score': round(1 - similarity, 4),
        'flagged'        : flagged,
        'verdict'        : 'Possible forgery' if flagged else 'Signatures match',
    }

if __name__ == '__main__':
    # Configuration
    DATASET_DIR = "D:/extra/signatures/train" # Set this to your local dataset folder
    SAVE_PATH = "models/siamese_signatures.pt" # Where to save the trained model

    print("Options:")
    print("1. Train Siamese Network")
    print("2. Test Inference (Needs 2 signature images)")
    
    choice = input("Enter choice (1/2): ")

    if choice == '1':
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        train_siamese(DATASET_DIR, SAVE_PATH, epochs=20, n_pairs=5000)
    elif choice == '2':
        img1_path = input("Enter path to Image 1: ")
        img2_path = input("Enter path to Image 2: ")

        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            res = compare_signatures(img1, img2, SAVE_PATH)
            print()
            for k, v in res.items():
                print(f"{k}: {v}")
        else:
            print("Invalid image paths.")
