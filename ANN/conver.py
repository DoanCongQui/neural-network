from PIL import Image
import os

# Thư mục chứa ảnh gốc
input_folder = "images/lisa"
# Thư mục lưu ảnh đã resize
output_folder = "out/lisa"

# Tạo thư mục lưu nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Duyệt từng ảnh trong thư mục
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        # Mở ảnh
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Resize về 128x128
        img_resized = img.resize((128, 128))

        # Lưu ảnh
        save_path = os.path.join(output_folder, filename)
        img_resized.save(save_path)

print("Đã resize xong toàn bộ ảnh.")
