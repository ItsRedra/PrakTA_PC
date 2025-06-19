import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def extract_glcm_features(image_path):
    image = imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = rgb2gray(image)
    image = (image * 255).astype(np.uint8)
    # Use single distance and angle to reduce feature dimensionality
    distances = [1]
    angles = [0]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = {}
    for prop in props:
        feat = graycoprops(glcm, prop)[0, 0]
        features[prop] = feat
    return features

def load_dataset():
    features = []
    labels = []
    base_path = 'dataset'
    categories = ['narkoba', 'normal']
    for category in categories:
        folder_path = os.path.join(base_path, category)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                path = os.path.join(folder_path, filename)
                feat_dict = extract_glcm_features(path)
                features.append(list(feat_dict.values()))
                labels.append(category)
    return np.array(features), np.array(labels)

def train_knn_and_save():
    X, y = load_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

    param_grid = {'n_neighbors': list(range(1, 11))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    best_knn = grid.best_estimator_

    y_pred = best_knn.predict(X_test)

    print("Best k:", grid.best_params_)
    print("Akurasi Test:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Asli")
    plt.show()

    os.makedirs('model', exist_ok=True)
    joblib.dump(best_knn, 'model/knn_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(le, 'model/label_encoder.pkl')

if __name__ == "__main__":
    train_knn_and_save()
