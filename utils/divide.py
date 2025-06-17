import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Шляхи
src_dir = "dataset/mixed_train"
dst_dir = "dataset/mixed_valid"
csv_path = os.path.join(src_dir, "labels.csv")

# Завантажити CSV
df = pd.read_csv(csv_path)

# Розбити 80/20
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Перемістити файли для validation
os.makedirs(dst_dir, exist_ok=True)
for fname in valid_df['filename']:
    src = os.path.join(src_dir, fname)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(src):
        shutil.move(src, dst)

# Зберегти оновлені CSV
train_df.to_csv(os.path.join(src_dir, "labels.csv"), index=False)
valid_df.to_csv(os.path.join(dst_dir, "labels.csv"), index=False)
