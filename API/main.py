import io
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms, models
from PIL import Image
from google import genai

# --- 0. REGISTRASI ARSITEKTUR ViT-5 KE TIMM ---
VIT5_REPO_PATH = os.path.join(os.path.dirname(__file__), "ViT-5")
if os.path.exists(VIT5_REPO_PATH):
    sys.path.append(VIT5_REPO_PATH)

import timm
try:
    import models_vit5  # noqa: F401
    print("Modul models_vit5 berhasil di-import!")
except ImportError:
    print("Warning: Modul models_vit5 tidak ditemukan. Pastikan folder ViT-5 sudah di-clone.")

app = FastAPI(title="ViT-5 Diabetic Retinopathy Detection API with Fundus Validation")

# --- 1. KONFIGURASI GEMINI AI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "Silahkan masukkan GEMINI_API_KEY Anda di environment variable GEMINI_API_KEY"

if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    ai_client = None
    print("Warning: GEMINI_API_KEY tidak ditemukan. Fallback ke rekomendasi statis.")

# --- 2. KONFIGURASI DEVICE & PATH MODEL ---
VIT5_IMAGE_SIZE = 224
VALIDATION_IMAGE_SIZE = 192  # Sesuai dengan ukuran crop model validasi

VIT5_MODEL_PATH = "best_vit5_aptos.pth"
VALIDATOR_MODEL_PATH = "best_fundus_validator.pth"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Menggunakan GPU Mac (Metal / MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Menggunakan GPU CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Menggunakan CPU")

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(DEVICE)

# SEVERITY MAP
SEVERITY_MAP = {
    0: {
        "keparahan": "No DR (Normal)",
        "rekomendasi": "Mata Anda dalam kondisi sehat. Lakukan pemeriksaan rutin setiap 1 tahun sekali."
    },
    1: {
        "keparahan": "Mild (Ringan)",
        "rekomendasi": "Terdapat indikasi awal. Jaga kadar gula darah dan lakukan pemeriksaan kembali dalam 6-12 bulan."
    },
    2: {
        "keparahan": "Moderate (Sedang)",
        "rekomendasi": "Terdeteksi perubahan pada pembuluh darah retina. Konsultasikan dengan dokter spesialis mata."
    },
    3: {
        "keparahan": "Severe (Berat)",
        "rekomendasi": "Kerusakan retina cukup signifikan. Segera buat janji dengan dokter spesialis mata untuk penanganan medis."
    },
    4: {
        "keparahan": "Proliferative DR (Sangat Berat)",
        "rekomendasi": "Kondisi kritis. Diperlukan tindakan medis/terapi laser segera oleh dokter spesialis mata."
    }
}

# --- 3. FUNGSI LOAD MODEL ---
def load_validator_model():
    """Memuat model validasi MobileNetV3-Small (2 Kelas: 0=Non-Fundus, 1=Fundus)."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    
    if os.path.exists(VALIDATOR_MODEL_PATH):
        checkpoint = torch.load(VALIDATOR_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        print("Model Validasi Fundus berhasil di-load!")
        return model
    else:
        print(f"Warning: File {VALIDATOR_MODEL_PATH} tidak ditemukan.")
        return None

def load_vit5_model():
    """Memuat model deteksi keparahan ViT-5."""
    model = timm.create_model(
        'vit5_base',
        pretrained=False,
        num_classes=1,
        img_size=VIT5_IMAGE_SIZE
    )
    
    if not os.path.exists(VIT5_MODEL_PATH):
        print(f"Warning: File {VIT5_MODEL_PATH} tidak ditemukan.")
        return None

    checkpoint = torch.load(VIT5_MODEL_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_ema' in checkpoint:
            state_dict = checkpoint['model_ema']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {
        (key.replace("module.", "") if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"ViT-5 Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.to(DEVICE)
    model.eval()
    return model

# Inisialisasi Model
validator_model = load_validator_model()
vit5_model = load_vit5_model()

# --- 4. PREPROCESSING PIPELINES ---
def preprocess_for_validation(pil_img):
    """Pipeline preprocessing khusus untuk model validasi fundus (CenterCrop px)."""
    val_transform = transforms.Compose([
        transforms.Resize(200),
        transforms.CenterCrop(VALIDATION_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return val_transform(pil_img).unsqueeze(0)

def resize_image(img, image_size):
    return cv2.resize(img, (image_size, image_size))

def image_filter(img, image_size):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel_enhanced = clahe.apply(l_channel)
    
    lab_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    img_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return img_clahe

def preprocess_for_vit5(image_bytes, image_size):
    """Pipeline preprocessing khusus untuk model ViT-5 (CLAHE + Normalize)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Gambar tidak valid atau terkorupsi")

    img_resized = resize_image(img, image_size)
    img_processed = image_filter(img_resized, image_size)
    img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(img_rgb).unsqueeze(0)

# --- 5. HELPER GEMINI AI & CONFIDENCE ---
def get_gemini_recommendation(keparahan_label: str) -> str:
    if not ai_client:
        return None

    prompt = (
        f"Berikan rekomendasi penanganan medis langsung kepada pasien "
        f"yang baru saja mendapatkan hasil deteksi Retinopati Diabetik: '{keparahan_label}'.\n\n"
        f"Aturan output:\n"
        f"1. Tulis langsung kalimat rekomendasi dari dokter untuk pasien dalam 2-3 kalimat.\n"
        f"2. Sebutkan tindakan awal/gaya hidup yang perlu dilakukan pasien serta saran apakah harus konsultasi ke dokter spesialis mata (Sp.M).\n"
        f"3. DILARANG menggunakan kata pengganti 'Anda' yang merujuk ke dokter (gunakan sudut pandang dokter yang berbicara ke pasien).\n"
        f"4. DILARANG menggunakan pemformatan teks seperti cetak tebal (bold), cetak miring (italic), atau tanda bintang (*).\n"
        f"5. DILARANG menambahkan kalimat pembuka atau penutup."
    )

    try:
        response = ai_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gagal memanggil Gemini API: {e}")
        return None

def compute_pseudo_confidence(raw_output: float, predicted_class: int) -> float:
    distance = abs(raw_output - predicted_class)
    confidence = 1 - min(distance / 0.5, 1.0)
    return round(confidence, 4)

# --- 6. ENDPOINT API ---
@app.post("/api/deteksi")
async def detect_retinopathy(file: UploadFile = File(...)):
    if vit5_model is None:
        raise HTTPException(status_code=500, detail="Model ViT-5 gagal di-load di server.")
        
    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # =========================================================
        # TAHAP 1: VALIDASI CITRA FUNDUS (MobileNetV3-Small)
        # =========================================================
        if validator_model is not None:
            val_tensor = preprocess_for_validation(pil_img).to(DEVICE)
            
            with torch.no_grad():
                val_output = validator_model(val_tensor)
                val_probs = torch.softmax(val_output, dim=1)
                val_pred = torch.argmax(val_probs, dim=1).item()
                fundus_confidence = round(val_probs[0][1].item(), 4)
                fundus_prob = val_probs[0][1].item()

            # Jika terdeteksi BUKAN citra fundus (kelas 0)
            # if val_pred == 0:
            if fundus_prob < 0.85:
                return {
                    "is_fundus": False,
                    "tingkat_keparahan": f"Tidak Valid ({fundus_prob:.4f})",
                    "rekomendasi_penanganan": "File yang diunggah bukan citra fundus mata yang valid. Silakan unggah citra fundus yang jelas dan sesuai.",
                }

        # =========================================================
        # TAHAP 2: DETEKSI KEPARAHAN RETINOPATI DIABETIK (ViT-5)
        # =========================================================
        input_tensor = preprocess_for_vit5(image_bytes, VIT5_IMAGE_SIZE).to(DEVICE)

        with torch.no_grad():
            input_tensor = input_tensor.to(DEVICE, dtype=torch.float32)
            outputs = vit5_model(input_tensor)
            
            if outputs.shape[1] == 1:
                raw_output = torch.clamp(outputs, 0, 4).item()
                pred_index = int(round(raw_output))
                confidence_score = compute_pseudo_confidence(raw_output, pred_index)
            else:
                probs = torch.softmax(outputs, dim=1)
                confidence_score = round(torch.max(probs).item(), 4)
                _, predicted = torch.max(outputs, 1)
                pred_index = predicted.item()

        result_data = SEVERITY_MAP.get(pred_index, SEVERITY_MAP[0])
        keparahan_str = result_data["keparahan"]

        # Rekomendasi AI dari Gemini
        rekomendasi_ai = get_gemini_recommendation(keparahan_str)
        rekomendasi_final = rekomendasi_ai if rekomendasi_ai else result_data["rekomendasi"]

        return {
            "is_fundus": True,
            "tingkat_keparahan": keparahan_str,
            "rekomendasi_penanganan": rekomendasi_final,
            "confidence_score": confidence_score
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        import traceback
        print("ERROR DETAIL:", traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Gagal memproses gambar: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)