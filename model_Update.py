import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from imblearn.over_sampling import RandomOverSampler
from tkinter import Tk, filedialog, Button, Label, Frame

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class DepressionDetectionModel:
    def __init__(self, img_size=(64, 64), sequence_length=10, batch_size=4, epochs=30):
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.class_names = ['No Depression', 'Depression']
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.CoarseDropout(max_holes=5, p=0.2)
        ])
        self.casme_path = 'CASME2_compressed'
        self.ckplus_path = 'ckextended.csv'

    def _load_ckplus(self):
        data = pd.read_csv(self.ckplus_path)
        depression_emotions = [0, 2, 4]
        X, y = [], []
        for _, row in data.iterrows():
            label = 1 if row['emotion'] in depression_emotions else 0
            pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
            img = pixels.reshape((48, 48))
            img = cv2.resize(img, self.img_size)
            img = img.astype('float32') / 255.0
            X.append(img)
            y.append(label)
        return np.array(X), np.array(y)

    def evaluate_separately(self, X_test, y_test, source_labels):
        print("\nEvaluating by dataset source...")
        sources = np.unique(source_labels)
        for source in sources:
            mask = source_labels == source
            print(f"\nEvaluation for {source}:")
            self.evaluate(X_test[mask], y_test[mask])

    def visualize_class_distribution(self, labels):
        sns.countplot(x=labels)
        plt.title("Class Distribution")
        plt.xticks(ticks=[0, 1], labels=self.class_names)
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.show()

    def load_data_sequences(self, xlsx_path='CASME2-coding-20140508.xlsx'):
        df = pd.read_excel(xlsx_path)
        depressive_emotions = ['sadness', 'fear', 'disgust']
        sequences, labels, sources = [], [], []

        for _, row in df.iterrows():
            subject = row['Subject']
            folder = row['Filename']
            try:
                onset = int(row['OnsetFrame'])
                apex = int(row['ApexFrame'])
            except ValueError:
                continue

            emotion = str(row['Estimated Emotion']).lower()
            label = 1 if emotion in depressive_emotions else 0

            subject_path = os.path.join(self.casme_path, f'sub{int(subject):02d}', folder)
            if not os.path.exists(subject_path):
                continue
            img_files = sorted([f for f in os.listdir(subject_path) if f.endswith('.jpg')])
            frames = []
            for i in range(onset, apex + 1):
                if i - 1 < len(img_files):
                    img_path = os.path.join(subject_path, img_files[i - 1])
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, self.img_size)
                        img = img.astype('float32') / 255.0
                        frames.append(img)
            if len(frames) >= self.sequence_length:
                for j in range(0, len(frames) - self.sequence_length + 1):
                    clip = frames[j:j + self.sequence_length]
                    sequences.append(clip)
                    labels.append(label)
                    sources.append('CASME_II')

        ckplus_images, ckplus_labels = self._load_ckplus()
        for img, label in zip(ckplus_images, ckplus_labels):
            seq = np.repeat(img[np.newaxis, ...], self.sequence_length, axis=0)
            sequences.append(seq)
            labels.append(label)
            sources.append('CK+')

        sequences = np.array(sequences)
        labels = np.array(labels)
        sources = np.array(sources)

        print("Original label distribution:", np.bincount(labels))
        self.visualize_class_distribution(labels)

        X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
            sequences, labels, sources, test_size=0.2, stratify=labels, random_state=42
        )
        self.test_sources = sources_test

        flat_train = X_train.reshape(X_train.shape[0], -1)
        ros = RandomOverSampler(random_state=42)
        flat_train_res, y_train_res = ros.fit_resample(flat_train, y_train)
        X_train = flat_train_res.reshape(-1, self.sequence_length, self.img_size[0], self.img_size[1])

        print("After oversampling (train set):", np.bincount(y_train_res))
        self.visualize_class_distribution(y_train_res)

        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train_res)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_res)
        self.class_weights = {cls: weight for cls, weight in zip(classes, weights)}

        return (X_train, y_train_res), (X_test, y_test)

    def build_model(self):
        print("Building 3D CNN model...")
        input_shape = (self.sequence_length, self.img_size[0], self.img_size[1], 1)
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling3D((1, 2, 2))(x)
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)
        x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        self.model = models.Model(inputs, outputs)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        print(self.model.summary())
        return self.model

    def train(self, X_train, y_train, X_val, y_val):
        print("Training model...")
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ModelCheckpoint('best_model.h5', monitor='val_auc', save_best_only=True, mode='max'),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        self.model.fit(
            X_train[..., np.newaxis], y_train,
            validation_data=(X_val[..., np.newaxis], y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            class_weight=self.class_weights,
            callbacks=callbacks_list,
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        X_test = X_test[..., np.newaxis]
        y_prob = self.model.predict(X_test)

        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_threshold = thresholds[np.argmax(f1)]
        self.best_threshold = best_threshold  # store best threshold
        print(f"Best Threshold (F1): {best_threshold:.2f}")

        y_pred = (y_prob > best_threshold).astype(int)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def visualize_activations(self, sequence):
        input_sequence = np.expand_dims(sequence, axis=0)[..., np.newaxis]

        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv3D):
                last_conv_layer = layer.name
                break
        if not last_conv_layer:
            print("No Conv3D layer found.")
            return

        grad_model = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(last_conv_layer).output, self.model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_sequence)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print("Gradient is None.")
            return

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

        center_frame = sequence[self.sequence_length // 2].squeeze()
        heatmap_resized = cv2.resize(heatmap[self.sequence_length // 2], (self.img_size[1], self.img_size[0]))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(center_frame, cmap='gray')
        axes[0].set_title("Original Frame")
        axes[0].axis('off')

        axes[1].imshow(center_frame, cmap='gray')
        axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        axes[1].set_title("Activation Heatmap")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        threshold = getattr(self, "best_threshold", 0.5)
        print("Prediction:", "Depressed" if predictions[0][0] > threshold else "Not Depressed", f"(Confidence: {predictions[0][0]:.2f})")

    def predict_from_image(self, use_webcam=False):
        if use_webcam:
            print("Capturing image from webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Webcam could not be opened.")
                return
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print("Failed to capture image.")
                return
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            file_path = filedialog.askopenfilename(title='Select an Image File', filetypes=[("Image files", "*.jpg *.png *.jpeg")])
            if not file_path:
                print("No file selected.")
                return
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Failed to load image.")
                return

        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        sequence = np.repeat(img[np.newaxis, ...], self.sequence_length, axis=0)

        self.visualize_activations(sequence)

def launch_gui(detector):
    def on_upload():
        detector.predict_from_image(use_webcam=False)

    def on_webcam():
        detector.predict_from_image(use_webcam=True)

    root = Tk()
    root.title("Depression Detection GUI")

    frame = Frame(root)
    frame.pack(pady=20)

    Label(frame, text="Choose input method for prediction:").pack(pady=10)
    Button(frame, text="Upload Image", command=on_upload, width=20).pack(pady=5)
    Button(frame, text="Use Webcam", command=on_webcam, width=20).pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    detector = DepressionDetectionModel()
    (X_train, y_train), (X_test, y_test) = detector.load_data_sequences()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    detector.build_model()
    detector.train(X_train, y_train, X_val, y_val)
    detector.evaluate_separately(X_test, y_test, detector.test_sources)

    sample_idx = np.random.randint(len(X_test))
    detector.visualize_activations(X_test[sample_idx])

    launch_gui(detector)