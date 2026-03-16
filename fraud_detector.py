# DOCUMENT FRAUD DETECTION

import os, io, re, cv2, sys, json, time, argparse, random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# CONFIG

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR  = SCRIPT_DIR / 'models'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

AE_THRESHOLD_CHEQUE  = 0.018
AE_THRESHOLD_AADHAAR = 0.020
ELA_FLAG_THRESHOLD   = 0.04
SCORE_GENUINE        = 0.28
SCORE_SUSPICIOUS     = 0.50


# MODEL ARCHITECTURES

class DocumentAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),   nn.Sigmoid(),
        )
    def forward(self, x): return self.decoder(self.encoder(x))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        backbone    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        for p in backbone.parameters(): p.requires_grad = False
        self.backbone   = backbone
        self.comparator = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward_one(self, x): return self.backbone(x)
    def forward(self, a, b):
        return self.comparator(torch.abs(self.forward_one(a) - self.forward_one(b)))


# MODEL LOADING

_cache = {}

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_autoencoder(doc_type):
    key = f'ae_{doc_type}'
    if key in _cache: return _cache[key]
    path = MODEL_DIR / f'autoencoder_{doc_type}.pt'
    if not path.exists(): print(f'  ⚠️  Autoencoder not found: {path}'); return None
    m = DocumentAutoencoder()
    m.load_state_dict(torch.load(str(path), map_location='cpu'))
    m.eval(); _cache[key] = m
    print(f'  ✅ Autoencoder loaded ({doc_type})')
    return m

def load_siamese():
    if 'siamese' in _cache: return _cache['siamese']
    path = MODEL_DIR / 'siamese_signatures.pt'
    if not path.exists(): print(f'  ⚠️  Siamese not found'); return None
    m = SiameseNetwork()
    m.load_state_dict(torch.load(str(path), map_location='cpu'))
    m.eval(); _cache['siamese'] = m
    print(f'  ✅ Siamese loaded')
    return m

def load_xgboost(doc_type):
    key = f'xgb_{doc_type}'
    if key in _cache: return _cache[key]
    path = MODEL_DIR / f'xgboost_{doc_type}.pkl'
    if not path.exists(): return None
    try:
        import joblib, xgboost
        m = joblib.load(str(path)); _cache[key] = m
        print(f'  ✅ XGBoost loaded ({doc_type})')
        return m
    except Exception as e:
        print(f'  ⚠️  XGBoost load failed: {e}'); return None


# IMAGE UTILITIES

def load_image(file_path):
    path = Path(file_path)
    if path.suffix.lower() == '.pdf':
        from pdf2image import convert_from_path
        pil = convert_from_path(str(path), dpi=200)[0].convert('RGB')
    else:
        img = cv2.imread(str(path))
        if img is None: raise ValueError(f'Cannot read: {path}')
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(pil), pil

def find_images(folder):
    exts = {'.jpg','.jpeg','.png','.bmp','.tiff','.tif'}
    return [str(f) for f in Path(folder).rglob('*')
            if f.is_file() and f.suffix.lower() in exts]


# MODULE 1 — DOCUMENT TYPE

def detect_doc_type(image_rgb, force=None):
    if force in ('cheque','aadhaar'): return force, 'FORCED'
    h, w  = image_rgb.shape[:2]
    gray  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    ratio = w / h
    cs = 0.0; as_ = 0.0

    if ratio > 2.0:   cs += 0.40
    elif ratio > 1.6: cs += 0.25
    elif ratio > 1.3: cs += 0.10

    bottom = gray[int(h*0.85):, :]
    if 0.02 < np.sum(bottom<80)/bottom.size < 0.35 and np.sum(bottom>200)/bottom.size > 0.45:
        cs += 0.30
    if np.sum(gray > 200) / gray.size > 0.55: cs += 0.15

    top_hsv  = hsv[:int(h*0.20), :]
    top_blue = cv2.inRange(top_hsv, (100,50,50), (130,255,255))
    tbr      = np.sum(top_blue > 0) / top_blue.size
    if tbr < 0.04: cs += 0.10

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                             minLineLength=w//3, maxLineGap=20)
    horiz = 0
    if lines is not None:
        for ln in lines:
            x1,y1,x2,y2 = ln[0]
            if abs(np.arctan2(y2-y1,x2-x1)*180/np.pi) < 5: horiz += 1
    if horiz >= 3: cs += 0.10

    if tbr > 0.10:   as_ += 0.40
    elif tbr > 0.04: as_ += 0.20
    if ratio < 1.6:  as_ += 0.20
    try:
        from pyzbar.pyzbar import decode
        if any(q.type in ('QRCODE','PDF417') for q in decode(image_rgb)): as_ += 0.35
    except: pass
    photo = gray[int(h*0.15):int(h*0.70), int(w*0.02):int(w*0.35)]
    if float(np.std(photo)) > 40: as_ += 0.15

    if cs > as_ + 0.15:   return 'cheque',  'HIGH' if cs  > 0.6 else 'MEDIUM'
    elif as_ > cs + 0.15: return 'aadhaar', 'HIGH' if as_ > 0.6 else 'MEDIUM'
    else:                 return ('cheque' if ratio > 1.5 else 'aadhaar'), 'LOW'


# MODULE 2 — ELA

def run_ela(image_pil, quality=90):
    buf = io.BytesIO()
    img_rgb = image_pil.convert('RGB')
    img_rgb.save(buf, format='JPEG', quality=quality); buf.seek(0)
    recomp = Image.open(buf).convert('RGB')
    orig   = np.array(img_rgb).astype(np.float32)
    recomp = np.array(recomp).astype(np.float32)
    diff   = np.abs(orig - recomp)
    ela_display = np.clip(diff * 10, 0, 255).astype(np.uint8)
    ela_gray    = cv2.cvtColor(ela_display, cv2.COLOR_RGB2GRAY)
    raw_score   = float(np.mean(diff)) / 255.0
    return ela_gray, round(raw_score, 6), raw_score > ELA_FLAG_THRESHOLD

def find_tampered_regions(ela_gray, threshold=80):
    _, thresh = cv2.threshold(ela_gray, threshold, 255, cv2.THRESH_BINARY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    dilated   = cv2.dilate(thresh, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []; h, w = ela_gray.shape
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500: continue
        x, y, bw, bh = cv2.boundingRect(c)
        regions.append({'x':x,'y':y,'w':bw,'h':bh,'area_pct':round(area/(h*w)*100,2)})
    return sorted(regions, key=lambda r: -r['area_pct'])

def compute_ela_score(ela_raw, ela_region_count):
    score_from_value   = min(ela_raw / 0.06, 1.0)
    score_from_regions = 0.0 if ela_raw < 0.03 else min(ela_region_count / 40.0, 1.0)
    return round(max(score_from_value, score_from_regions), 4)


# MODULE 3 — CHEQUE RULES

def check_cheque(image_rgb):
    h, w = image_rgb.shape[:2]; flags = []; scores = []

    # MICR band
    micr      = image_rgb[int(h*0.85):, :]
    micr_gray = cv2.cvtColor(micr, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(micr_gray, 248, 255, cv2.THRESH_BINARY)
    bright_pct = np.sum(bright > 0) / bright.size
    mean_b     = float(np.mean(micr_gray))
    if bright_pct > 0.45 and mean_b > 220:
        flags.append(f'⚠️  MICR band {bright_pct*100:.1f}% bright — possible whitener')
        scores.append(0.70)
    elif bright_pct > 0.60:
        flags.append(f'⚠️  MICR band high brightness ({bright_pct*100:.1f}%)')
        scores.append(0.40)
    else: scores.append(0.0)

    # Correction fluid
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, vwhite = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY)
    white_pct = np.sum(vwhite > 0) / vwhite.size
    if white_pct > 0.55:
        flags.append(f'⚠️  Excessive white ({white_pct*100:.1f}%) — possible erasure')
        scores.append(0.45)
    else: scores.append(0.0)

    # ELA on amount
    _, ela_s, ela_flagged = run_ela(Image.fromarray(image_rgb[int(h*0.5):int(h*0.75), int(w*0.5):]))
    if ela_flagged:
        flags.append(f'⚠️  Amount region ELA={ela_s:.4f} — possible tampering')
        scores.append(min(ela_s / 0.06, 0.8))
    else: scores.append(0.0)

    # OCR amount check
    try:
        from paddleocr import PaddleOCR
        ocr    = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(image_rgb[int(h*0.4):, :], cls=True)
        text   = ' '.join(line[1][0] for r in (result or [[]])
                          for line in (r or []) if line and len(line)>1).lower()
        numbers = re.findall(r'[\d,]+(?:\.\d{2})?', text.replace(' ',''))
        amounts = []
        for n in numbers:
            try:
                v = float(n.replace(',',''))
                if v > 100: amounts.append(v)
            except: pass
        if len(amounts) >= 2 and max(amounts)/max(min(amounts),1) > 10:
            flags.append(f'⚠️  Amount mismatch: {sorted(amounts)}')
            scores.append(0.55)
        else: scores.append(0.0)
    except: scores.append(0.0)

    # MICR ink consistency
    micr_std = float(np.std(micr_gray))
    if micr_std > 90:
        flags.append(f'⚠️  MICR inconsistent ink (std={micr_std:.1f})')
        scores.append(0.35)
    else: scores.append(0.0)

    final = max(scores) if scores else 0.0
    if not flags: flags.append('✅ No cheque-specific fraud detected')
    return {'cheque_score':round(final,4),'flags':flags,
            'bright_pct':round(bright_pct,4),'white_pct':round(white_pct,4)}


# MODULE 4 — AADHAAR RULES

def check_aadhaar(image_rgb):
    h, w = image_rgb.shape[:2]; flags = []; scores = []; qr_found = False
    try:
        from pyzbar.pyzbar import decode
        for qr in decode(Image.fromarray(image_rgb)):
            if qr.type in ('QRCODE','PDF417'):
                qr_found = True
                content  = qr.data.decode('utf-8', errors='ignore')
                if 'uid' in content.lower() or len(content) > 50:
                    flags.append('✅ QR code found and valid'); scores.append(0.0)
                else:
                    flags.append('⚠️  QR found but content looks wrong'); scores.append(0.35)
                break
        if not qr_found:
            flags.append('⚠️  No QR code — genuine Aadhaar always has one'); scores.append(0.60)
    except Exception as e:
        flags.append(f'⚠️  QR check failed: {str(e)[:40]}'); scores.append(0.25)

    try:
        from paddleocr import PaddleOCR
        ocr    = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(image_rgb, cls=True)
        text   = ' '.join(line[1][0] for r in (result or [[]])
                          for line in (r or []) if line and len(line)>1)
        if re.findall(r'\d{4}\s*[-\s]\s*\d{4}\s*[-\s]\s*\d{4}', text):
            flags.append('✅ Aadhaar number format found'); scores.append(0.0)
        else:
            flags.append('⚠️  Aadhaar number pattern not found'); scores.append(0.30)
    except: scores.append(0.0)

    top_hsv  = cv2.cvtColor(image_rgb[:int(h*0.15),:], cv2.COLOR_RGB2HSV)
    blue_pct = np.sum(cv2.inRange(top_hsv,(100,50,50),(130,255,255))>0) / (top_hsv.shape[0]*top_hsv.shape[1])
    if blue_pct < 0.05:
        flags.append('⚠️  Missing UIDAI blue header'); scores.append(0.30)
    else:
        flags.append(f'✅ UIDAI blue header ({blue_pct*100:.1f}%)'); scores.append(0.0)

    _, ela_s, ela_flagged = run_ela(Image.fromarray(image_rgb[:int(h*0.6), int(w*0.2):]))
    if ela_flagged:
        flags.append(f'⚠️  Name/DOB ELA={ela_s:.4f}'); scores.append(min(ela_s/0.06, 0.8))
    else:
        flags.append('✅ Name/DOB region looks unedited'); scores.append(0.0)

    return {'aadhaar_score':round(max(scores) if scores else 0.0, 4),
            'flags':flags,'qr_found':qr_found}


# MODULE 5 — AUTOENCODER

ae_transform = T.Compose([T.Resize((128,128)), T.ToTensor()])

def detect_cheque_background(image_rgb):

    h, w   = image_rgb.shape[:2]
    # Sample center region
    center = image_rgb[int(h*0.3):int(h*0.7), int(w*0.1):int(w*0.9)]
    hsv    = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)

    # Check dominant hue
    green_mask  = cv2.inRange(hsv, (35,30,100),  (85,255,255))
    blue_mask   = cv2.inRange(hsv, (90,30,100),  (130,255,255))
    yellow_mask = cv2.inRange(hsv, (20,50,100),  (35,255,255))

    total = center.shape[0] * center.shape[1]
    green_pct  = np.sum(green_mask > 0)  / total
    blue_pct   = np.sum(blue_mask > 0)   / total
    yellow_pct = np.sum(yellow_mask > 0) / total

    if green_pct  > 0.15: return 'green'
    if blue_pct   > 0.15: return 'blue'
    if yellow_pct > 0.15: return 'yellow'
    return 'white'

def run_autoencoder(image_pil, doc_type):
    model = load_autoencoder(doc_type)
    if model is None:
        return {'ae_score':0.0,'raw_error':0.0,'flagged':False,
                'reason':'Model not found'}

    tensor = ae_transform(image_pil.convert('RGB')).unsqueeze(0)
    with torch.no_grad(): recon = model(tensor)
    raw_error = float(torch.mean((tensor - recon) ** 2))

    # Dynamic threshold based on background color
    if doc_type == 'cheque':
        image_rgb = np.array(image_pil.convert('RGB'))
        bg_color  = detect_cheque_background(image_rgb)
        if bg_color == 'white':
            threshold = 0.018
        elif bg_color in ('green', 'blue', 'yellow'):
            threshold = 0.045
        else:
            threshold = 0.025
    else:
        threshold = AE_THRESHOLD_AADHAAR

    ae_score = min(raw_error / (threshold * 3), 1.0)
    flagged  = raw_error > threshold

    return {
        'ae_score' : round(ae_score, 4),
        'raw_error': round(raw_error, 6),
        'flagged'  : flagged,
        'reason'   : f'Visual anomaly (bg={bg_color if doc_type=="cheque" else "n/a"})' if flagged
                     else 'Document reconstructs normally',
    }


# MODULE 6 — SIGNATURE (generalised — ALL Indian bank cheques)

sig_transform = T.Compose([
    T.Resize((224,224)), T.Grayscale(num_output_channels=3),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_signature_region(image_rgb, doc_type='cheque'):

    if doc_type != 'cheque': return None, None
    h, w = image_rgb.shape[:2]

    zones = {
        'right_bottom'  : image_rgb[int(h*0.62):int(h*0.90), int(w*0.60):int(w*0.99)],
        'right_mid'     : image_rgb[int(h*0.55):int(h*0.85), int(w*0.55):int(w*0.99)],
        'center_bottom' : image_rgb[int(h*0.62):int(h*0.90), int(w*0.35):int(w*0.80)],
        'full_bottom'   : image_rgb[int(h*0.65):int(h*0.92), int(w*0.05):int(w*0.99)],
        'right_wide'    : image_rgb[int(h*0.58):int(h*0.88), int(w*0.45):int(w*0.99)],
        'lower_third'   : image_rgb[int(h*0.70):int(h*0.88), int(w*0.50):int(w*0.99)],
    }

    best_zone = None; best_name = None; best_score = -1
    for name, zone in zones.items():
        if zone.size == 0: continue
        gray = cv2.cvtColor(zone, cv2.COLOR_RGB2GRAY)
        _, dark = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        lines_mask = cv2.bitwise_or(
            cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk),
            cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk))
        sig_only   = cv2.subtract(dark, lines_mask)
        ink_score  = np.sum(sig_only > 0) / max(sig_only.size, 1)
        if ink_score > best_score:
            best_score = ink_score; best_zone = zone; best_name = name

    if best_zone is None: return None, None
    rh, rw = best_zone.shape[:2]; pad = 10
    inner  = best_zone[pad:rh-pad, pad:rw-pad]
    return Image.fromarray(inner if inner.size > 0 else best_zone), best_name


def check_signature_present(sig_array):

    gray = cv2.cvtColor(sig_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    pad  = 15
    gray = gray[pad:max(h-pad,pad+1), pad:max(w-pad,pad+1)]
    h, w = gray.shape

    # Check ink
    _, dark   = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    ink_ratio = np.sum(dark > 0) / max(dark.size, 1)

    # Remove straight lines (box borders, ruled lines)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    sig_only = cv2.subtract(dark,
               cv2.bitwise_or(cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk),
                              cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)))

    cleaned = cv2.morphologyEx(sig_only,
              cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sig_strokes = 0; total_area = 0; x_pos = []; y_pos = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 60: continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw / max(bh,1) > 8: continue
        if bh < 3: continue
        p = cv2.arcLength(c, True)
        if p == 0: continue
        if (4 * np.pi * area) / (p ** 2) > 0.03:
            sig_strokes += 1; total_area += area
            x_pos.append(x + bw//2); y_pos.append(y + bh//2)

    sig_ink_ratio = np.sum(sig_only > 0) / max(sig_only.size, 1)
    x_spread = (max(x_pos)-min(x_pos))/max(w,1) if len(x_pos)>=2 else 0.0

    print(f'        DEBUG  ink={ink_ratio:.5f}  sig_ink={sig_ink_ratio:.5f}'
          f'  strokes={sig_strokes}  spread={x_spread:.3f}')

    # 3-tier detection
    has_flowing = sig_strokes >= 4 and x_spread > 0.25
    has_compact = sig_ink_ratio > 0.005 and sig_strokes >= 2
    has_any_ink = ink_ratio > 0.010 and sig_strokes >= 2

    if has_flowing or has_compact or has_any_ink:
        psc = 0.0 if sig_strokes >= 3 else 0.20
        return {'present':True,'ink_ratio':round(ink_ratio,5),
                'sig_ink_ratio':round(sig_ink_ratio,5),
                'stroke_count':sig_strokes,'x_spread':round(x_spread,3),
                'presence_score':psc}
    else:
        return {'present':False,'ink_ratio':round(ink_ratio,5),
                'sig_ink_ratio':round(sig_ink_ratio,5),
                'stroke_count':sig_strokes,'x_spread':round(x_spread,3),
                'presence_score':0.85}


def compare_signatures(sig_pil, ref_pil):
    model = load_siamese()
    if model is None:
        return {'sig_score':0.0,'similarity':0.0,'flagged':False,'verdict':'No model'}
    device = get_device(); model = model.to(device)
    t1 = sig_transform(sig_pil).unsqueeze(0).to(device)
    t2 = sig_transform(ref_pil).unsqueeze(0).to(device)
    with torch.no_grad(): similarity = model(t1, t2).item()
    flagged = similarity < 0.45
    return {'sig_score':round(1-similarity,4),'similarity':round(similarity,4),
            'flagged':flagged,
            'verdict':'Possible forgery' if flagged else 'Signatures match'}


# FUSION

def fuse_scores(ela_s, rule_s, ae_s, sig_s, doc_type):
    xgb = load_xgboost(doc_type)
    if xgb is not None:
        prob = float(xgb.predict_proba(np.array([[ela_s,rule_s,ae_s,sig_s]]))[0][1])
        return round(prob,4), 'XGBoost'

    sw  = 0.15 if sig_s > 0.0 else 0.0
    rem = 1.0 - sw
    if rule_s > 0.5:
        w = {'ela':0.20*rem,'rule':0.55*rem,'ae':0.25*rem,'sig':sw}
    elif ela_s > 0.5:
        w = {'ela':0.45*rem,'rule':0.30*rem,'ae':0.25*rem,'sig':sw}
    elif ae_s > 0.5:
        w = {'ela':0.25*rem,'rule':0.30*rem,'ae':0.45*rem,'sig':sw}
    else:
        w = {'ela':0.30*rem,'rule':0.35*rem,'ae':0.35*rem,'sig':sw}

    score = ela_s*w['ela'] + rule_s*w['rule'] + ae_s*w['ae'] + sig_s*w['sig']
    max_s = max(ela_s, rule_s, ae_s, sig_s)
    if max_s > 0.75: score = max(score, max_s * 0.72)
    return round(score,4), 'Weighted'


def score_to_verdict(score):
    if score < SCORE_GENUINE:    return 'GENUINE','✅'
    elif score < SCORE_SUSPICIOUS: return 'REVIEW','🟡'
    else:                          return 'FRAUD','🚨'


# MAIN PIPELINE

def analyze_document(image_path, force_type=None, ref_sig_path=None, verbose=True):
    if verbose:
        print(); print('═'*60)
        print('  DOCUMENT FRAUD ANALYSIS'); print('═'*60)
        print(f'  File: {Path(image_path).name}'); print()

    try: image_rgb, image_pil = load_image(image_path)
    except Exception as e: print(f'  ❌ {e}'); return None

    h, w = image_rgb.shape[:2]
    if verbose: print(f'  Resolution : {w} × {h}')

    doc_type, confidence = detect_doc_type(image_rgb, force=force_type)
    if verbose: print(f'  Doc type   : {doc_type.upper()} ({confidence})'); print()

    # ELA
    if verbose: print('  [1/4] ELA — pixel tampering...')
    ela_gray, ela_raw, ela_flagged = run_ela(image_pil)
    ela_regions = find_tampered_regions(ela_gray)
    ela_s       = compute_ela_score(ela_raw, len(ela_regions))
    if verbose:
        tag = '🚨' if ela_flagged else '✅'
        print(f'        {tag}  raw={ela_raw:.5f}  score={ela_s:.4f}  ({len(ela_regions)} regions)')

    # Rules
    if verbose: print(f'  [2/4] {doc_type.upper()} rules...')
    if doc_type == 'cheque':
        doc_result = check_cheque(image_rgb); rule_s = doc_result['cheque_score']
    else:
        doc_result = check_aadhaar(image_rgb); rule_s = doc_result['aadhaar_score']
    if verbose:
        print(f'        {"🚨" if rule_s>0.4 else "✅"}  score={rule_s:.4f}')
        for flag in doc_result.get('flags',[]): print(f'             {flag}')

    # Autoencoder
    if verbose: print('  [3/4] Autoencoder — visual anomaly...')
    ae_result = run_autoencoder(image_pil, doc_type)
    ae_s      = ae_result['ae_score']
    if verbose:
        print(f'        {"🚨" if ae_result["flagged"] else "✅"}  score={ae_s:.4f}'
              f'  raw={ae_result["raw_error"]:.6f}')
        print(f'             {ae_result["reason"]}')

    # Signature
    if verbose: print('  [4/4] Signature check...')
    sig_s = 0.0; sig_result = {}; no_sig_found = False

    if doc_type == 'cheque':
        sig_crop, zone_name = extract_signature_region(image_rgb, doc_type)
        if sig_crop is None:
            if verbose: print('        ⚠️  Could not crop signature region')
            sig_s = 0.20
        else:
            if verbose: print(f'        ℹ️  Best zone: {zone_name}')
            presence = check_signature_present(np.array(sig_crop))

            if not presence['present']:
                sig_s = 0.85; no_sig_found = True
                sig_result = {'sig_score':sig_s,'verdict':'NO SIGNATURE FOUND'}
                if verbose:
                    print(f'        🚨 NO SIGNATURE FOUND IN FIELD')
                    print(f'             ink={presence["ink_ratio"]:.5f}'
                          f'  strokes={presence["stroke_count"]}'
                          f'  spread={presence["x_spread"]:.3f}')
                    print(f'             A cheque without a signature is INVALID')

            elif presence['presence_score'] > 0:
                sig_s = presence['presence_score']
                sig_result = {'sig_score':sig_s,'verdict':'FAINT SIGNATURE'}
                if verbose:
                    print(f'        ⚠️  FAINT/BORDERLINE SIGNATURE')
                    print(f'             ink={presence["ink_ratio"]:.5f}'
                          f'  strokes={presence["stroke_count"]}'
                          f'  spread={presence["x_spread"]:.3f}')
            else:
                if verbose:
                    print(f'        ✅  Signature present'
                          f'  (ink={presence["ink_ratio"]:.4f}'
                          f'  strokes={presence["stroke_count"]}'
                          f'  spread={presence["x_spread"]:.3f})')
                if ref_sig_path:
                    try:
                        ref_pil    = Image.open(ref_sig_path).convert('RGB')
                        sig_result = compare_signatures(sig_crop, ref_pil)
                        sig_s      = sig_result['sig_score']
                        tag        = '🚨' if sig_result['flagged'] else '✅'
                        if verbose:
                            print(f'        {tag}  {sig_result["verdict"]}'
                                  f'  similarity={sig_result["similarity"]:.4f}')
                    except Exception as e:
                        if verbose: print(f'        ⚠️  Comparison failed: {e}')
                else:
                    if verbose:
                        print('        ℹ️  No reference — presence confirmed only')
                        print('             Use --ref_sig to check for forgery')
    else:
        if verbose: print('        ⏭️  Not applicable for Aadhaar')

    # Per-region ELA
    region_ela = {}
    try:
        crop_map = (
            {'amount_figures' : image_rgb[int(h*0.53):int(h*0.70), int(w*0.55):int(w*0.95)],
             'amount_words'   : image_rgb[int(h*0.38):int(h*0.55), int(w*0.05):int(w*0.90)],
             'payee_field'    : image_rgb[int(h*0.20):int(h*0.40), int(w*0.05):int(w*0.90)],
             'date_field'     : image_rgb[int(h*0.02):int(h*0.20), int(w*0.65):int(w*0.98)],
             'signature_field': image_rgb[int(h*0.60):int(h*0.90), int(w*0.50):int(w*0.99)],
             'micr_band'      : image_rgb[int(h*0.85):, :]}
            if doc_type == 'cheque' else
            {'photo_region': image_rgb[int(h*0.15):int(h*0.70), int(w*0.02):int(w*0.35)],
             'top_strip'   : image_rgb[:int(h*0.15), :]}
        )
        for rname, rimg in crop_map.items():
            if rimg.size > 0:
                _, rs, _ = run_ela(Image.fromarray(rimg))
                region_ela[rname] = rs
    except Exception: pass

    # Missing signature check
    if no_sig_found:
        if verbose:
            print(); print('  🚨 HARD RULE: No signature → forcing FRAUD')
        final_score = 0.85; method = 'Hard Rule (No Signature)'; verdict = 'FRAUD'
    else:
        final_score, method = fuse_scores(ela_s, rule_s, ae_s, sig_s, doc_type)
        verdict, _          = score_to_verdict(final_score)

    results = {
        'file':'','doc_type':doc_type,'dt_confidence':confidence,
        'ela_score':ela_s,'ela_raw':ela_raw,'ela_regions':len(ela_regions),
        'rule_score':rule_s,'ae_score':ae_s,'ae_raw':ae_result['raw_error'],
        'sig_score':sig_s,'final_score':final_score,
        'risk_pct':round(final_score*100,1),'verdict':verdict,'method':method,
        'rule_flags':doc_result.get('flags',[]),
        'ela_gray':ela_gray,'ela_regions_list':ela_regions,
        'image_rgb':image_rgb,'region_ela':region_ela,
        'ae_result':ae_result,'sig_result':sig_result,
    }
    results['file'] = str(image_path)

    if verbose: _print_report(results); _show_visual(results)
    return results


def _print_report(r):
    print(); print('═'*60); print('  FINAL REPORT'); print('═'*60)
    print(f'  Doc type     : {r["doc_type"].upper()}  ({r["dt_confidence"]})')
    print(f'  ELA score    : {r["ela_score"]:.4f}  ({r["ela_regions"]} regions, raw={r["ela_raw"]:.5f})')
    print(f'  Rule score   : {r["rule_score"]:.4f}')
    print(f'  AE score     : {r["ae_score"]:.4f}  (raw={r["ae_raw"]:.6f})')
    print(f'  Sig score    : {r["sig_score"]:.4f}')
    print(f'  {"─"*44}')
    print(f'  RISK SCORE   : {r["risk_pct"]:.1f}%  ({r["method"]})')
    print()
    colors = {'GENUINE':'\033[92m','REVIEW':'\033[93m','FRAUD':'\033[91m'}
    icon   = {'GENUINE':'✅','REVIEW':'🟡','FRAUD':'🚨'}.get(r['verdict'],'⚪')
    print(f'  VERDICT      : {colors.get(r["verdict"],"")}{r["verdict"]}\033[0m  {icon}')
    print('═'*60)
    if r.get('region_ela'):
        print(); print('  Per-region ELA:')
        for rname, rscore in sorted(r['region_ela'].items(), key=lambda x: -x[1]):
            bar  = '█' * int(rscore/0.005)
            flag = ' ← SUSPICIOUS' if rscore > ELA_FLAG_THRESHOLD else ''
            print(f'    {rname:20} {rscore:.5f}  {bar[:30]}{flag}')
    if r.get('rule_flags'):
        print(); print('  Flags:')
        for f in r['rule_flags']: print(f'    {f}')
    # Gemini report
    print(); print('═'*60)
    print('  GEMINI AI ANALYSIS')
    print('═'*60)
    gemini_report = generate_gemini_report(r)
    print(gemini_report)
    print('═'*60)


def _show_visual(r):
    try:
        vc = {'GENUINE':'#27ae60','REVIEW':'#e67e22','FRAUD':'#e74c3c'}
        vcolor = vc.get(r['verdict'],'#7f8c8d')

        fig, axes = plt.subplots(1, 3, figsize=(22,7))
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes: ax.set_facecolor('#16213e')

        axes[0].imshow(r['image_rgb'])
        axes[0].set_title('Original Document', fontweight='bold', fontsize=12, color='white', pad=10)
        axes[0].axis('off')

        axes[1].imshow(r['ela_gray'], cmap='hot')
        axes[1].set_title('ELA Map  (bright = tampered)', fontweight='bold', fontsize=12, color='white', pad=10)
        axes[1].axis('off')

        axes[2].imshow(r['image_rgb'])
        for i, reg in enumerate(r['ela_regions_list'][:6]):
            axes[2].add_patch(patches.Rectangle(
                (reg['x'],reg['y']),reg['w'],reg['h'],
                linewidth=2,edgecolor='#e74c3c',facecolor='none',alpha=0.9))
            axes[2].text(reg['x']+3,reg['y']-6,f"{i+1}",
                        color='#e74c3c',fontsize=9,fontweight='bold')
        axes[2].set_title('Suspicious Regions', fontweight='bold', fontsize=12, color='white', pad=10)
        axes[2].axis('off')

        # Risk score bar
        score  = r['risk_pct'] / 100
        bar_ax = fig.add_axes([0.1, 0.02, 0.8, 0.04])
        bar_ax.set_xlim(0,1); bar_ax.set_ylim(0,1); bar_ax.set_facecolor('#2d3436')
        bar_ax.barh(0.5, 0.28, height=0.8, color='#27ae60', alpha=0.4)
        bar_ax.barh(0.5, 0.22, height=0.8, color='#e67e22', alpha=0.4, left=0.28)
        bar_ax.barh(0.5, 0.50, height=0.8, color='#e74c3c', alpha=0.4, left=0.50)
        bar_ax.axvline(score, color='white', linewidth=3)
        bar_ax.text(score, 1.3, f'{r["risk_pct"]:.1f}%', ha='center', va='bottom',
                   color='white', fontsize=11, fontweight='bold')
        bar_ax.text(0.14,0.5,'GENUINE',ha='center',va='center',color='#27ae60',fontsize=8,fontweight='bold')
        bar_ax.text(0.39,0.5,'REVIEW', ha='center',va='center',color='#e67e22',fontsize=8,fontweight='bold')
        bar_ax.text(0.75,0.5,'FRAUD',  ha='center',va='center',color='#e74c3c',fontsize=8,fontweight='bold')
        bar_ax.axis('off')

        fig.suptitle(
            f'{r["verdict"]}  ·  Risk: {r["risk_pct"]:.1f}%  ·  '
            f'{r["doc_type"].upper()}  ·  {r["method"]}',
            fontsize=15, fontweight='bold', color=vcolor, y=0.98)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.12, wspace=0.05)

        ts       = int(time.time())
        savepath = OUTPUT_DIR / f'report_{Path(r["file"]).stem}_{ts}.png'
        plt.savefig(str(savepath), dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
        print(f'\n  📊 Report → {savepath}')
        plt.show()
    except Exception as e:
        print(f'  ⚠️  Visualization skipped: {e}')


# XGBOOST TRAINING

def collect_xgboost_scores(genuine_folder, output_csv):
    import csv
    from tqdm import tqdm

    genuine_images = find_images(genuine_folder)
    print(f'Found {len(genuine_images)} genuine images')
    if not genuine_images: print('❌ No images'); return None

    rows = []

    print('\nProcessing genuine images...')
    for img_path in tqdm(genuine_images, desc='Genuine'):
        try:
            image_rgb, image_pil = load_image(img_path)
            ela_gray, ela_raw, _ = run_ela(image_pil)
            ela_s    = compute_ela_score(ela_raw, len(find_tampered_regions(ela_gray)))
            rule_s   = check_cheque(image_rgb)['cheque_score']
            ae_s     = run_autoencoder(image_pil, 'cheque')['ae_score']
            sig_crop, _ = extract_signature_region(image_rgb, 'cheque')
            sig_s    = 0.0 if (sig_crop and check_signature_present(np.array(sig_crop))['present']) else 0.85
            rows.append({'ela_score':ela_s,'rule_score':rule_s,'ae_score':ae_s,'sig_score':sig_s,'label':0})
        except Exception as e: print(f'  Skipped: {e}')

    print('\nGenerating tampered versions...')
    for img_path in tqdm(genuine_images, desc='Tampered'):
        try:
            image_rgb, _ = load_image(img_path)
            h, w = image_rgb.shape[:2]
            for ttype, (x1,y1,x2,y2) in [
                ('amount_figures', (int(w*.55),int(h*.53),int(w*.95),int(h*.70))),
                ('amount_words',   (int(w*.05),int(h*.38),int(w*.90),int(h*.55))),
                ('payee_name',     (int(w*.05),int(h*.20),int(w*.90),int(h*.40))),
                ('date',           (int(w*.65),int(h*.02),int(w*.98),int(h*.20))),
            ]:
                t = image_rgb.copy()
                cv2.rectangle(t,(x1,y1),(x2,y2),(255,255,255),-1)
                cv2.putText(t,str(np.random.randint(10000,9999999)),
                            (x1+10,y2-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
                t_pil = Image.fromarray(t)
                ela_gray, ela_raw, _ = run_ela(t_pil)
                ela_s = compute_ela_score(ela_raw, len(find_tampered_regions(ela_gray)))
                rule_s = check_cheque(t)['cheque_score']
                ae_s   = run_autoencoder(t_pil, 'cheque')['ae_score']
                rows.append({'ela_score':ela_s,'rule_score':rule_s,'ae_score':ae_s,'sig_score':0.0,'label':1})
        except Exception as e: print(f'  Skipped: {e}')

    with open(output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ela_score','rule_score','ae_score','sig_score','label'])
        w.writeheader(); w.writerows(rows)

    g = sum(1 for r in rows if r['label']==0); t = len(rows)-g
    print(f'\n✅ CSV saved → {output_csv}')
    print(f'   Genuine: {g}  Tampered: {t}  Total: {len(rows)}')
    return output_csv


def train_xgboost(csv_path, doc_type='cheque'):
    import pandas as pd, joblib
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    df = pd.read_csv(csv_path)
    print(f'Rows: {len(df)}  Genuine: {len(df[df.label==0])}  Fraud: {len(df[df.label==1])}')
    X = df[['ela_score','rule_score','ae_score','sig_score']].values
    y = df['label'].values
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    model = XGBClassifier(
        n_estimators=200,max_depth=4,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.8,
        scale_pos_weight=len(y_tr[y_tr==0])/max(len(y_tr[y_tr==1]),1),
        eval_metric='logloss',random_state=42,verbosity=0)
    model.fit(X_tr,y_tr,eval_set=[(X_te,y_te)],verbose=False)

    y_pred = model.predict(X_te)
    print(f'\nAccuracy: {accuracy_score(y_te,y_pred)*100:.1f}%')
    print(classification_report(y_te,y_pred,target_names=['Genuine','Fraud']))

    feats = ['ela_score','rule_score','ae_score','sig_score']
    print('Feature importance:')
    for f, imp in sorted(zip(feats,model.feature_importances_),key=lambda x:-x[1]):
        print(f'  {f:12} {"█"*int(imp*40)} {imp*100:.1f}%')

    save_path = MODEL_DIR / f'xgboost_{doc_type}.pkl'
    joblib.dump(model, str(save_path))
    print(f'\n✅ XGBoost saved → {save_path}')
    return model


# BATCH ANALYZE

def batch_analyze(folder_path, force_type=None, output_csv=None):
    from tqdm import tqdm; import csv
    files = find_images(folder_path)
    if not files: print(f'❌ No images in: {folder_path}'); return []
    print(f'Found {len(files)} documents'); rows = []
    for fp in tqdm(files, desc='Analyzing'):
        try:
            r = analyze_document(fp, force_type=force_type, verbose=False)
            if r: rows.append({'filename':Path(fp).name,'doc_type':r['doc_type'],
                               'ela_score':r['ela_score'],'rule_score':r['rule_score'],
                               'ae_score':r['ae_score'],'sig_score':r['sig_score'],
                               'risk_pct':r['risk_pct'],'verdict':r['verdict'],'method':r['method']})
        except Exception as e: print(f'\n  ⚠️  Skipped {Path(fp).name}: {e}')

    print(); print('═'*62); print('  BATCH SUMMARY'); print('═'*62)
    fn = sum(1 for r in rows if r['verdict']=='FRAUD')
    rn = sum(1 for r in rows if r['verdict']=='REVIEW')
    gn = sum(1 for r in rows if r['verdict']=='GENUINE')
    print(f'  Total: {len(rows)}  ✅ {gn}  🟡 {rn}  🚨 {fn}'); print('═'*62)
    for row in sorted(rows,key=lambda x:-x['risk_pct']):
        icon = {'FRAUD':'🚨','REVIEW':'🟡','GENUINE':'✅'}.get(row['verdict'],'⚪')
        print(f'  {row["filename"][:31]:<32} {row["doc_type"]:<8} {row["risk_pct"]:>5.1f}%  {icon} {row["verdict"]}')
    if output_csv and rows:
        with open(output_csv,'w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print(f'\n  📄 CSV → {output_csv}')
    return rows
    #---------------------------------------------------------------------
def generate_gemini_report(results):

    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()

        # Paste API key
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Build region ELA summary
        region_text = '\n'.join(
            f'  {k:20} {v:.5f} {"← SUSPICIOUS" if v > 0.04 else ""}'
            for k, v in sorted(
                results.get('region_ela', {}).items(),
                key=lambda x: -x[1]
            )
        )

        # Rule flags
        flags_text = '\n'.join(
            f'  {f}' for f in results.get('rule_flags', [])
        )

        sig_verdict = results.get('sig_result', {}).get('verdict', 'Not checked')

        prompt = f"""
You are a senior bank fraud analyst in India. Analyze this document 
check result and write a clear, professional report for a bank officer.

Document Type  : {results['doc_type'].upper()}
Final Verdict  : {results['verdict']}
Risk Score     : {results['risk_pct']}%
Detection Method: {results['method']}

Module Scores:
  ELA score (pixel tampering)  : {results['ela_score']:.4f}  ({results['ela_regions']} suspicious regions)
  Rules score (document checks): {results['rule_score']:.4f}
  Autoencoder score (anomaly)  : {results['ae_score']:.4f}
  Signature score              : {results['sig_score']:.4f}

Rule flags:
{flags_text if flags_text else '  No flags'}

Per-region ELA scores (higher = more tampered):
{region_text if region_text else '  Not available'}

Signature status: {sig_verdict}

Write a professional 4-5 sentence fraud analysis report.
Rules:
- Start directly with the verdict (GENUINE / FRAUD / REVIEW NEEDED)
- Mention exactly which fields look suspicious and why
- Explain what the ELA score means in plain language
- End with a clear recommendation (process / reject / escalate)
- Do NOT use bullet points — write in paragraphs
- Keep it under 150 words
- Write as if reporting to a bank branch manager
"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f'(Gemini report unavailable: {e})'

# CLI

def main():
    parser = argparse.ArgumentParser(description='Document Fraud Detector — All Indian Banks')
    parser.add_argument('--image',          type=str)
    parser.add_argument('--folder',         type=str)
    parser.add_argument('--type',           type=str, choices=['cheque','aadhaar'])
    parser.add_argument('--ref_sig',        type=str)
    parser.add_argument('--csv',            type=str)
    parser.add_argument('--train_xgb',      action='store_true')
    parser.add_argument('--genuine_folder', type=str)
    args = parser.parse_args()

    print()
    print('╔══════════════════════════════════════════════╗')
    print('║  Document Fraud Detector — All Indian Banks  ║')
    print('╚══════════════════════════════════════════════╝')
    print(f'  Models : {MODEL_DIR}')
    print(f'  Outputs: {OUTPUT_DIR}')
    print(f'  Device : {get_device()}')
    print()

    if args.train_xgb:
        if not args.genuine_folder:
            print('❌ --genuine_folder required'); sys.exit(1)
        csv_path = args.csv or str(MODEL_DIR / 'xgb_scores.csv')
        collect_xgboost_scores(args.genuine_folder, csv_path)
        train_xgboost(csv_path, doc_type=args.type or 'cheque')

    elif args.image:
        analyze_document(image_path=args.image, force_type=args.type,
                         ref_sig_path=args.ref_sig, verbose=True)

    elif args.folder:
        batch_analyze(folder_path=args.folder, force_type=args.type,
                      output_csv=args.csv or str(OUTPUT_DIR/'batch_results.csv'))
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python fraud_detector.py --image cheque.jpg')
        print('  python fraud_detector.py --image cheque.jpg --ref_sig sig.jpg')
        print('  python fraud_detector.py --folder ./docs/ --csv results.csv')
        print('  python fraud_detector.py --train_xgb --genuine_folder ./genuine/')


if __name__ == '__main__':
    main()