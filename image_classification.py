import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Dataset path
DATA_PATH = "f:/临时任务/分类任务/dataset2"
IMG_SIZE = (128, 128)  # Resize images to this size

def load_and_analyze_data():
    """
    Load data, analyze class distribution and visualize
    """
    # Get all image files
    image_files = glob.glob(os.path.join(DATA_PATH, "*.jpg"))
    
    # Extract class from filename
    labels = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        if filename.startswith("cloudy"):
            labels.append("cloudy")
        elif filename.startswith("shine"):
            labels.append("shine")
        else:
            labels.append("unknown")
    
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts, color=['gray', 'yellow'])
    plt.title("Class Distribution")
    plt.xlabel("Weather Class")
    plt.ylabel("Number of Images")
    
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.savefig("class_distribution.png")
    plt.show()
    
    print(f"Total images: {len(image_files)}")
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count} images")
    
    return image_files, labels

def preprocess_data(image_files, labels):
    """
    Preprocess images and extract features
    """
    # Initialize data containers
    X_raw = []  # Original images resized
    X_hog = []  # HOG features
    X_color = []  # Color histograms
    y = []
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Process each image
    for i, img_path in enumerate(image_files):
        try:
            # Load and resize image
            img = Image.open(img_path)
            img = img.resize(IMG_SIZE)
            img_array = np.array(img)
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]  # Remove alpha channel
                
            X_raw.append(img_array)
            
            # Extract HOG features
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hog_features = extract_hog_features(gray)
            X_hog.append(hog_features)
            
            # Extract color histogram features
            color_features = extract_color_histogram(img_array)
            X_color.append(color_features)
            
            y.append(encoded_labels[i])
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X_raw = np.array(X_raw)
    X_hog = np.array(X_hog)
    X_color = np.array(X_color)
    y = np.array(y)
    
    # Normalize pixel values for raw images
    X_raw = X_raw / 255.0
    
    # Combine HOG and color features for ML models
    X_features = np.hstack((X_hog, X_color))
    
    # Split data into train and test sets
    X_raw_train, X_raw_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        X_raw, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert labels to categorical for CNN
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    print(f"Feature extraction complete.")
    print(f"Raw image data shape: {X_raw.shape}")
    print(f"HOG features shape: {X_hog.shape}")
    print(f"Color features shape: {X_color.shape}")
    print(f"Combined features shape: {X_features.shape}")
    
    return (X_raw_train, X_raw_test, X_feat_train, X_feat_test, 
            y_train, y_test, y_train_cat, y_test_cat, label_encoder)

def extract_hog_features(image):
    """
    Extract Histogram of Oriented Gradients features
    """
    # Calculate HOG features
    win_size = (64, 64)
    cell_size = (8, 8)
    block_size = (16, 16)
    nbins = 9
    
    # Resize image to match window size
    resized = cv2.resize(image, win_size)
    
    # Calculate HOG features
    hog = cv2.HOGDescriptor(win_size, block_size, cell_size, cell_size, nbins)
    features = hog.compute(resized)
    
    return features.flatten()

def extract_color_histogram(image, bins=32):
    """
    Extract color histogram features
    """
    # Convert to HSV color space for better color representation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculate histograms for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
    
    # Combine histograms
    hist_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    
    return hist_features

def train_random_forest(X_train, X_test, y_train, y_test, label_encoder):

    print("\n--- Training Random Forest model ---")
    
    # Handle imbalanced data
    class_weights = {
        i: len(y_train) / (len(np.unique(y_train)) * np.sum(y_train == i))
        for i in np.unique(y_train)
    }
    
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weights,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predict
    y_pred = rf_model.predict(X_test)
    
    # Evaluate
    evaluate_model(y_test, y_pred, label_encoder, 'Random Forest')
    
    return rf_model

def train_svm(X_train, X_test, y_train, y_test, label_encoder):

    print("\n--- Training SVM model ---")
    
    # Handle imbalanced data
    class_weights = {
        i: len(y_train) / (len(np.unique(y_train)) * np.sum(y_train == i))
        for i in np.unique(y_train)
    }
    
    # Train model
    svm_model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight=class_weights,
        random_state=42
    )
    
    svm_model.fit(X_train, y_train)
    
    # Predict
    y_pred = svm_model.predict(X_test)
    
    # Evaluate
    evaluate_model(y_test, y_pred, label_encoder, 'SVM')
    
    return svm_model

def create_cnn_model(input_shape, num_classes):

    model = Sequential([
        # Convolutional layers
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_cnn(X_train, X_test, y_train_cat, y_test_cat, y_test, label_encoder):
    """
    Train and evaluate a CNN model
    """
    print("\n--- Training CNN model ---")
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_test))
    
    # Create model
    cnn_model = create_cnn_model(input_shape, num_classes)
    cnn_model.summary()
    
    # Handle class imbalance with class weights
    unique_classes = np.unique(np.argmax(y_train_cat, axis=1))
    class_weights = {
        i: len(y_train_cat) / (len(unique_classes) * np.sum(np.argmax(y_train_cat, axis=1) == i))
        for i in unique_classes
    }
    
    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = cnn_model.fit(
        X_train, y_train_cat,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stop]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    plt.show()
    
    # Predict
    y_pred_prob = cnn_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Evaluate
    evaluate_model(y_test, y_pred, label_encoder, 'CNN')
    
    return cnn_model

def evaluate_model(y_true, y_pred, label_encoder, model_name):
    """
    Evaluate model performance and visualize results
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_names = label_encoder.classes_
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"{model_name} Classification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()

def predict_and_visualize(models, X_test, y_test, X_raw_test, label_encoder, n_samples=5):
    """
    Visualize predictions from different models
    """
    rf_model, svm_model, cnn_model = models
    class_names = label_encoder.classes_
    
    # Randomly select samples (at least one from each class if possible)
    indices = []
    for class_idx in range(len(class_names)):
        class_indices = np.where(y_test == class_idx)[0]
        if len(class_indices) > 0:
            indices.extend(np.random.choice(class_indices, 
                                          min(1, len(class_indices)), 
                                          replace=False))
    
    # Add more random samples if needed
    if len(indices) < n_samples:
        remaining = n_samples - len(indices)
        remaining_indices = np.setdiff1d(np.arange(len(y_test)), indices)
        if len(remaining_indices) > 0:
            indices.extend(np.random.choice(remaining_indices, 
                                          min(remaining, len(remaining_indices)),
                                          replace=False))
    
    # Make predictions
    rf_preds = rf_model.predict(X_test[indices])
    svm_preds = svm_model.predict(X_test[indices])
    
    cnn_pred_probs = cnn_model.predict(X_raw_test[indices])
    cnn_preds = np.argmax(cnn_pred_probs, axis=1)
    
    # Visualize predictions
    plt.figure(figsize=(15, 3 * len(indices)))
    
    for i, idx in enumerate(indices):
        img = X_raw_test[idx]
        true_label = label_encoder.inverse_transform([y_test[idx]])[0]
        
        rf_pred = label_encoder.inverse_transform([rf_preds[i]])[0]
        svm_pred = label_encoder.inverse_transform([svm_preds[i]])[0]
        cnn_pred = label_encoder.inverse_transform([cnn_preds[i]])[0]
        
        plt.subplot(len(indices), 4, i * 4 + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}")
        plt.axis('off')
        
        plt.subplot(len(indices), 4, i * 4 + 2)
        plt.bar(class_names, rf_model.predict_proba(X_test[idx:idx+1])[0])
        plt.title(f"RF Pred: {rf_pred}")
        plt.ylim(0, 1)
        
        plt.subplot(len(indices), 4, i * 4 + 3)
        plt.bar(class_names, svm_model.predict_proba(X_test[idx:idx+1])[0])
        plt.title(f"SVM Pred: {svm_pred}")
        plt.ylim(0, 1)
        
        plt.subplot(len(indices), 4, i * 4 + 4)
        plt.bar(class_names, cnn_pred_probs[i])
        plt.title(f"CNN Pred: {cnn_pred}")
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()

def main():
    """
    Main function to run the image classification pipeline
    """
    print("=== Weather Image Classification ===")
    
    # 1. Load and analyze data
    print("\nStep 1: Loading and analyzing data...")
    image_files, labels = load_and_analyze_data()
    
    # 2. Preprocess data
    print("\nStep 2: Preprocessing data and extracting features...")
    (X_raw_train, X_raw_test, X_feat_train, X_feat_test, 
     y_train, y_test, y_train_cat, y_test_cat, label_encoder) = preprocess_data(image_files, labels)
    
    # 3. Train and evaluate models
    print("\nStep 3: Training and evaluating models...")
    
    # Random Forest
    rf_model = train_random_forest(X_feat_train, X_feat_test, y_train, y_test, label_encoder)
    
    # SVM
    svm_model = train_svm(X_feat_train, X_feat_test, y_train, y_test, label_encoder)
    
    # CNN
    cnn_model = train_cnn(X_raw_train, X_raw_test, y_train_cat, y_test_cat, y_test, label_encoder)
    
    # 4. Visualize predictions
    print("\nStep 4: Visualizing predictions...")
    predict_and_visualize(
        (rf_model, svm_model, cnn_model),
        X_feat_test, y_test, X_raw_test,
        label_encoder
    )
    
    print("\nClassification pipeline completed!")

if __name__ == "__main__":
    main() 