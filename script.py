import os, sys, time, requests, csv
from PIL import Image
from io import BytesIO
import torch
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim

# ── CONFIG ───────────────────────────────────────────────
BASE_URL = "https://api.genlook.app/tryon/v1"
API_KEY = "gk_8r9005iErw4VgqoHyw-11k3dXsDlwgPXo6GdbpQUU3s"

PRODUCT_TXT = "products.txt"
IMAGE_FOLDER = "test_images"
OUTPUT_FOLDER = "outputs"
CSV_FILE = "results.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── SESSION ──────────────────────────────────────────────
session = requests.Session()
session.headers["x-api-key"] = API_KEY

# ── LOAD CLIP ────────────────────────────────────────────
print("🔄 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP loaded\n")


# ── UTIL: Load image ─────────────────────────────────────
def load_image(src):
    if src.startswith("http"):
        return Image.open(BytesIO(requests.get(src).content)).convert("RGB")
    return Image.open(src).convert("RGB")


# ── SCORING ──────────────────────────────────────────────
def clip_score(img1, img2):
    inputs = processor(images=[img1, img2], return_tensors="pt", padding=True)
    features = model.get_image_features(**inputs)
    return torch.nn.functional.cosine_similarity(features[0], features[1], dim=0).item()


def ssim_score(img1, img2):
    img1 = np.array(img1.resize((256, 256)))
    img2 = np.array(img2.resize((256, 256)))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    score, _ = ssim(gray1, gray2, full=True)
    return score


def alignment_score(img):
    img = np.array(img)
    h, w, _ = img.shape
    center = img[h//4:3*h//4, w//4:3*w//4]
    variance = np.var(center)
    return min(variance / 5000, 1.0)


def combined_score(product_src, result_src):
    product_img = load_image(product_src)
    result_img = load_image(result_src)

    c = clip_score(product_img, result_img)
    s = ssim_score(product_img, result_img)
    a = alignment_score(result_img)

    final = (0.5 * c) + (0.3 * s) + (0.2 * a)

    return c, s, a, final


# ── API CALLS ────────────────────────────────────────────
def create_product(image_url, external_id):
    r = session.post(f"{BASE_URL}/products", json={
        "externalId": external_id,
        "title": "Batch Product",
        "description": "Auto-generated",
        "imageUrls": [image_url],
    })
    r.raise_for_status()
    return r.json()["externalId"]


def upload_photo(path):
    with open(path, "rb") as f:
        r = session.post(
            f"{BASE_URL}/images/upload",
            files={"file": (os.path.basename(path), f, "image/jpeg")},
        )
    r.raise_for_status()
    return r.json()["imageId"]


def generate_tryon(product_id, image_id):
    r = session.post(f"{BASE_URL}/try-on", json={
        "productId": product_id,
        "customerImageId": image_id,
    })
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        return None
    return data["generationId"]


def poll(gen_id, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        r = session.get(f"{BASE_URL}/generations/{gen_id}")
        r.raise_for_status()
        data = r.json()

        if data["status"] == "COMPLETED":
            return data
        if data["status"] == "FAILED":
            return None

        time.sleep(2)
    return None


# ── LOAD INPUTS ──────────────────────────────────────────
def load_products(txt):
    with open(txt, "r") as f:
        return [l.strip() for l in f if l.strip()]


def load_images(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


# ── SAVE IMAGE ───────────────────────────────────────────
def save_image(url, path):
    img = requests.get(url).content
    with open(path, "wb") as f:
        f.write(img)


# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":

    products = load_products(PRODUCT_TXT)
    images = load_images(IMAGE_FOLDER)

    print(f"🛍️ Products: {len(products)}")
    print(f"👤 Images: {len(images)}\n")

    # CSV header
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "product_url", "image_path", "result_url",
            "clip", "ssim", "alignment", "final_score"
        ])

    for i, product_url in enumerate(products):

        print(f"\n Product {i+1}")
        product_id = create_product(product_url, f"product-{i}")

        for j, img_path in enumerate(images):

            print(f"  👤 Image {j+1}")

            try:
                image_id = upload_photo(img_path)
                gen_id = generate_tryon(product_id, image_id)

                if not gen_id:
                    print("    ❌ Failed start")
                    continue

                result = poll(gen_id)

                if not result:
                    print("    ❌ Failed gen")
                    continue

                result_url = result["resultImageUrl"]

                # ── SCORING ──
                c, s, a, final = combined_score(product_url, result_url)

                print(f"    📊 Score: {final:.3f}")

                # ── SAVE IMAGE ──
                filename = f"p{i}_img{j}.jpg"
                save_path = os.path.join(OUTPUT_FOLDER, filename)
                save_image(result_url, save_path)

                # ── SAVE CSV ──
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        product_url, img_path, result_url,
                        c, s, a, final
                    ])

            except Exception as e:
                print(f"    ⚠️ Error: {e}")