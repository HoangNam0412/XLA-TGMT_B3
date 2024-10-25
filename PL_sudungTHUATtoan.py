import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

# Bước 1: Chuẩn bị dữ liệu (giả sử thư mục cats và dogs đã có hình ảnh)
DATADIR = "DATADIR = "E:\\Python\\BienDoiAnh\\phanLoaiHA\\pythonProject1"
"  # Thay đổi thành đường dẫn thực tế
CATEGORIES = ["Cat", "Dog"]

# Resize tất cả hình ảnh về cùng kích thước
IMG_SIZE = 100


def create_dataset():
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # 0 for Cat, 1 for Dog
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append(new_array)
                labels.append(class_num)
            except Exception as e:
                pass
    return np.array(data), np.array(labels)


# Tạo dữ liệu từ hình ảnh
X, y = create_dataset()

# Reshape và chuẩn hóa dữ liệu
X = X.reshape(X.shape[0], -1)  # Flatten the images
X = X / 255.0  # Normalize

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 2: Khởi tạo các mô hình
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Bước 3: Huấn luyện và đánh giá các mô hình
results = {}

for name, model in models.items():
    start_time = time.time()  # Bắt đầu đo thời gian
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()  # Kết thúc đo thời gian

    # Đánh giá kết quả
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Time': end_time - start_time
    }

# In kết quả
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
