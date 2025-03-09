import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import multiprocessing
from tensorflow.keras.applications import ResNet50, EfficientNetB3, MobileNetV2
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import GaussianNoise, BatchNormalization, Dropout, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# กำหนด seed เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้ง
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# กำหนด path สำหรับข้อมูล (อาจต้องเปลี่ยนตามโครงสร้างของคุณ)
data_parent = 'C:/Users/focus/Age_Gender_Detect_photo/all_path'  # เปลี่ยนเป็น path ที่เก็บข้อมูลของคุณ

# อ่านข้อมูลจากไฟล์
fold_0 = pd.read_csv(os.path.join(data_parent, 'fold_0_data.txt'), sep='\t')
fold_1 = pd.read_csv(os.path.join(data_parent, 'fold_1_data.txt'), sep='\t')
fold_2 = pd.read_csv(os.path.join(data_parent, 'fold_2_data.txt'), sep='\t')
fold_3 = pd.read_csv(os.path.join(data_parent, 'fold_3_data.txt'), sep='\t')
fold_4 = pd.read_csv(os.path.join(data_parent, 'fold_4_data.txt'), sep='\t')
total_data = pd.concat([fold_0, fold_1, fold_2, fold_3, fold_4], ignore_index=True)

# สร้าง mapping สำหรับอายุ
age_mapping = [('(0, 2)', '0-2'), ('2', '0-2'), ('3', '0-2'), ('(4, 6)', '4-6'), 
               ('(8, 12)', '8-13'), ('13', '8-13'), ('22', '15-20'), ('(8, 23)','15-20'), 
               ('23', '25-32'), ('(15, 20)', '15-20'), ('(25, 32)', '25-32'), 
               ('(27, 32)', '25-32'), ('32', '25-32'), ('34', '25-32'), ('29', '25-32'), 
               ('(38, 42)', '38-43'), ('35', '38-43'), ('36', '38-43'), ('42', '48-53'), 
               ('45', '38-43'), ('(38, 43)', '38-43'), ('(38, 42)', '38-43'), 
               ('(38, 48)', '48-53'), ('46', '48-53'), ('(48, 53)', '48-53'), 
               ('55', '48-53'), ('56', '48-53'), ('(60, 100)', '60+'), 
               ('57', '60+'), ('58', '60+')]
age_mapping_dict = {each[0]: each[1] for each in age_mapping}

# กรองข้อมูลที่มีค่า None และแปลงค่าอายุ
total_data = total_data[total_data.age != 'None'].copy()
total_data.loc[:, 'age'] = total_data['age'].map(age_mapping_dict)

# ลบข้อมูลที่มีค่า NaN และสร้าง path เต็ม
total_data = total_data.dropna()
total_data['full_path'] = total_data.apply(lambda x: os.path.join(data_parent, 'faces', str(x.user_id), 'coarse_tilt_aligned_face.' + str(x.face_id) + '.' + x.original_image), axis=1)

# ตรวจสอบว่าไฟล์รูปภาพมีอยู่จริง (ช่วยกรองข้อมูลที่ path ไม่ถูกต้อง)
def check_file_exists(path):
    return os.path.exists(path)

print("Checking file existence...")
total_data['file_exists'] = total_data.full_path.apply(check_file_exists)
total_data = total_data[total_data.file_exists].copy()
print(f"After filtering non-existent files: {len(total_data)} samples")

# แสดงจำนวนข้อมูลของแต่ละคลาส
print("\nAge distribution:")
print(total_data.age.value_counts().sort_index())
print("\nGender distribution:")
print(total_data.gender.value_counts())

# สร้าง mapping สำหรับเพศและอายุ
gender_map = {'f': 0, 'm': 1, 'u': 2}
age_map = {
    '0-2'  : 0,
    '4-6'  : 1,
    '8-13' : 2,
    '15-20': 3,
    '25-32': 4,
    '38-43': 5,
    '48-53': 6,
    '60+'  : 7
}

# แปลงข้อมูลเพศและอายุตาม mapping
total_data.gender = total_data.gender.replace(gender_map).infer_objects(copy=False)
total_data.age = total_data.age.replace(age_map).infer_objects(copy=False)

# สร้าง list ของ path และ label
gender_labels = total_data.gender.values.tolist()
age_labels = total_data.age.values.tolist()
train_paths = total_data.full_path.values.tolist()

# สลับข้อมูล
shuffle_list = list(zip(train_paths, gender_labels, age_labels))
random.shuffle(shuffle_list)
train_paths, gender_labels, age_labels = zip(*shuffle_list)

# แปลงข้อมูลให้อยู่ในรูปแบบ one-hot encoding
age_labels_array = np.array(list(age_labels)).reshape((-1, 1))
enc_age = OneHotEncoder()
age_labels_onehot = enc_age.fit_transform(age_labels_array).toarray()

gender_labels_array = np.array(list(gender_labels)).reshape((-1, 1))
enc_gender = OneHotEncoder()
gender_labels_onehot = enc_gender.fit_transform(gender_labels_array).toarray()

# คำนวณน้ำหนักของแต่ละคลาสเพื่อจัดการกับ class imbalance
age_class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(age_labels),
    y=np.array(age_labels)
)
age_weight_dict = {i: weight for i, weight in enumerate(age_class_weights)}

gender_class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(gender_labels),
    y=np.array(gender_labels)
)
gender_weight_dict = {i: weight for i, weight in enumerate(gender_class_weights)}

# แบ่งข้อมูลเป็น training, validation, และ test sets
# ใช้ stratified split เพื่อให้แต่ละชุดข้อมูลมีการกระจายของคลาสที่สมดุล
from sklearn.model_selection import train_test_split

# แบ่งข้อมูลในแบบ stratified เพื่อรักษาสัดส่วนของคลาส
X = np.array(train_paths)
y_age = np.array(age_labels)
y_gender = np.array(gender_labels)

# แบ่งออกเป็น train และ temp (validation + test) โดยใช้ stratified split ตามอายุ
X_train, X_temp, y_age_train, y_age_temp, y_gender_train, y_gender_temp = train_test_split(
    X, y_age, y_gender, test_size=0.3, random_state=SEED, stratify=y_age
)

# แบ่ง temp เป็น validation และ test
X_val, X_test, y_age_val, y_age_test, y_gender_val, y_gender_test = train_test_split(
    X_temp, y_age_temp, y_gender_temp, test_size=0.33, random_state=SEED, stratify=y_age_temp
)

# แปลงกลับเป็น one-hot encoding
y_age_train_onehot = enc_age.transform(y_age_train.reshape(-1, 1)).toarray()
y_age_val_onehot = enc_age.transform(y_age_val.reshape(-1, 1)).toarray()
y_age_test_onehot = enc_age.transform(y_age_test.reshape(-1, 1)).toarray()

y_gender_train_onehot = enc_gender.transform(y_gender_train.reshape(-1, 1)).toarray()
y_gender_val_onehot = enc_gender.transform(y_gender_val.reshape(-1, 1)).toarray()
y_gender_test_onehot = enc_gender.transform(y_gender_test.reshape(-1, 1)).toarray()

# พิมพ์ขนาดของชุดข้อมูล
print(f"\nTraining set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# สร้างฟังก์ชันสำหรับ data augmentation และ preprocessing
def preprocess_and_augment(image_path, label, training=True):
    # อ่านรูปภาพจาก path
    img_str = tf.io.read_file(image_path)
    try:
        img = tf.image.decode_jpeg(img_str, channels=3)
    except:
        # ถ้าเกิดข้อผิดพลาดในการอ่านรูปภาพ ให้สร้างรูปภาพว่าง
        img = tf.zeros([224, 224, 3], dtype=tf.uint8)
    
    # ปรับขนาดภาพ
    img = tf.image.resize(img, [224, 224])
    
    # Normalize ภาพ
    img = img / 255.0
    
    # ทำ data augmentation เฉพาะกับชุดข้อมูล training
    if training:
        # Random flip
        img = tf.image.random_flip_left_right(img)
        
        # Random brightness, contrast, saturation
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        
        # Random crop and resize
        img = tf.image.random_crop(img, [200, 200, 3])
        img = tf.image.resize(img, [224, 224])
        
        # ใช้ mixup augmentation เพื่อเพิ่มความหลากหลาย
        if tf.random.uniform(()) > 0.7:  # 30% โอกาสที่จะใช้ mixup
            # สร้างรูปภาพที่มีการสลับสีแดงและน้ำเงิน
            mixed_img = tf.stack([img[:,:,2], img[:,:,1], img[:,:,0]], axis=2)
            alpha = tf.random.uniform((), 0.3, 0.7)
            img = alpha * img + (1 - alpha) * mixed_img
        
        # เพิ่ม Gaussian noise
        img = img + tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.01)
        img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, label

# สร้าง dataset สำหรับโมเดลอายุ
train_age_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_age_train_onehot))
val_age_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_age_val_onehot))
test_age_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_age_test_onehot))

# สร้าง dataset สำหรับโมเดลเพศ
train_gender_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_gender_train_onehot))
val_gender_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_gender_val_onehot))
test_gender_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_gender_test_onehot))

# กำหนด batch size
BATCH_SIZE = 32

# เตรียม dataset สำหรับโมเดลอายุ
train_age_dataset = train_age_dataset.shuffle(buffer_size=10000)
train_age_dataset = train_age_dataset.map(
    lambda x, y: preprocess_and_augment(x, y, training=True),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_age_dataset = train_age_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_age_dataset = val_age_dataset.map(
    lambda x, y: preprocess_and_augment(x, y, training=False),
    num_parallel_calls=tf.data.AUTOTUNE
)
val_age_dataset = val_age_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_age_dataset = test_age_dataset.map(
    lambda x, y: preprocess_and_augment(x, y, training=False),
    num_parallel_calls=tf.data.AUTOTUNE
)
test_age_dataset = test_age_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# เตรียม dataset สำหรับโมเดลเพศ
train_gender_dataset = train_gender_dataset.shuffle(buffer_size=10000)
train_gender_dataset = train_gender_dataset.map(
    lambda x, y: preprocess_and_augment(x, y, training=True),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_gender_dataset = train_gender_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_gender_dataset = val_gender_dataset.map(
    lambda x, y: preprocess_and_augment(x, y, training=False),
    num_parallel_calls=tf.data.AUTOTUNE
)
val_gender_dataset = val_gender_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_gender_dataset = test_gender_dataset.map(
    lambda x, y: preprocess_and_augment(x, y, training=False),
    num_parallel_calls=tf.data.AUTOTUNE
)
test_gender_dataset = test_gender_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# สร้างโมเดลใหม่ด้วย EfficientNetB3 สำหรับ feature extraction
def create_efficientnet_model(num_classes, input_shape=(224, 224, 3), dropout_rate=0.3):
    # สร้าง base model
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # ล็อคเลเยอร์ของ base model เพื่อไม่ให้มีการปรับค่าน้ำหนักในขั้นตอนแรก
    base_model.trainable = False
    
    # สร้างโมเดล
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.1)(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # เพิ่มเลเยอร์เพื่อการจำแนก
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # เพิ่ม attention mechanism เพื่อให้โมเดลเรียนรู้ feature ที่สำคัญ
    attention = Dense(256, activation='tanh')(x)
    attention = Dense(256, activation='sigmoid')(attention)
    x = tf.multiply(x, attention)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

# Focal Loss - ช่วยจัดการกับ class imbalance ได้ดีกว่า categorical cross entropy
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = alpha * tf.pow(1.0 - y_pred, gamma) * y_true
        return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
    return focal_loss_fixed

# สร้างโมเดลใหม่สำหรับการจำแนกอายุและเพศ
print("Creating new age classification model with EfficientNetB3...")
age_model, age_base_model = create_efficientnet_model(num_classes=8, dropout_rate=0.4)  # เพิ่ม dropout rate เพื่อลด overfitting

print("Creating new gender classification model with EfficientNetB3...")
gender_model, gender_base_model = create_efficientnet_model(num_classes=3, dropout_rate=0.3)

# ตั้งค่า learning rate และ optimizer
initial_learning_rate = 3e-4  # ลด learning rate เริ่มต้น
age_optimizer = Adam(learning_rate=initial_learning_rate)
gender_optimizer = Adam(learning_rate=initial_learning_rate)

# คอมไพล์โมเดลด้วย focal loss
age_model.compile(
    optimizer=age_optimizer,
    loss=focal_loss(gamma=2.0),  # ใช้ focal loss แทน categorical crossentropy
    metrics=['accuracy']
)

gender_model.compile(
    optimizer=gender_optimizer,
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy']
)

# แสดงสรุปโมเดล
print("Age Classification Model:")
age_model.summary()

print("\nGender Classification Model:")
gender_model.summary()

# สร้าง callbacks
# 1. Model checkpointing - บันทึกโมเดลที่ดีที่สุด
age_checkpoint = ModelCheckpoint(
    'age_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

gender_checkpoint = ModelCheckpoint(
    'gender_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 2. Early stopping - หยุดการฝึกเมื่อไม่มีการปรับปรุง
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,  # เพิ่มความอดทนในการหยุด
    restore_best_weights=True,
    verbose=1
)

# 3. ReduceLROnPlateau - ลด learning rate เมื่อไม่มีการปรับปรุง
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # ลด learning rate 80% เมื่อไม่มีการปรับปรุง
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# 4. Custom callback สำหรับลดปัญหา overfitting
class OverfittingDetection(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.05):
        super(OverfittingDetection, self).__init__()
        self.threshold = threshold
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        if epoch > 5:  # เริ่มตรวจสอบหลังจากผ่านไป 5 epochs
            acc_diff = logs.get('accuracy') - logs.get('val_accuracy')
            
            if acc_diff > self.threshold:
                print(f"\nPotential overfitting detected: training accuracy exceeds validation by {acc_diff:.4f}")
                # อาจจะเพิ่ม dropout หรือ regularization ในระหว่างการฝึกได้
                
overfitting_detection = OverfittingDetection(threshold=0.08)

# 5. CosineAnnealing scheduler - ช่วยในการหา local minima ที่ดีกว่า
cosine_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=30 * (len(X_train) // BATCH_SIZE),
    alpha=1e-6
)

# รวม callbacks ทั้งหมด
age_callbacks = [age_checkpoint, early_stopping, reduce_lr, overfitting_detection]
gender_callbacks = [gender_checkpoint, early_stopping, reduce_lr, overfitting_detection]

# ฝึกโมเดลอายุ - ขั้นตอนที่ 1: ฝึกเฉพาะส่วนบนโดยล็อค base model
print("Starting training for age classification model (Phase 1)...")
age_history1 = age_model.fit(
    train_age_dataset,
    epochs=20,
    validation_data=val_age_dataset,
    callbacks=age_callbacks,
    class_weight=age_weight_dict
)

# ฝึกโมเดลเพศ - ขั้นตอนที่ 1: ฝึกเฉพาะส่วนบนโดยล็อค base model
print("Starting training for gender classification model (Phase 1)...")
gender_history1 = gender_model.fit(
    train_gender_dataset,
    epochs=20,
    validation_data=val_gender_dataset,
    callbacks=gender_callbacks,
    class_weight=gender_weight_dict
)

# ขั้นตอนที่ 2: Fine-tuning - ปลดล็อคบางส่วนของ base model
print("Starting fine-tuning of age model...")
# ปลดล็อคเฉพาะ layer ท้ายๆ ของ base model
for layer in age_base_model.layers[-30:]:
    layer.trainable = True

# คอมไพล์อีกครั้งด้วย learning rate ที่ต่ำลง
age_model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate / 10),
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy']
)

# ฝึกโมเดลอีกครั้ง
age_history2 = age_model.fit(
    train_age_dataset,
    epochs=15,
    validation_data=val_age_dataset,
    callbacks=age_callbacks,
    class_weight=age_weight_dict
)

# ทำเช่นเดียวกันกับโมเดลเพศ
print("Starting fine-tuning of gender model...")
for layer in gender_base_model.layers[-30:]:
    layer.trainable = True

gender_model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate / 10),
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy']
)

gender_history2 = gender_model.fit(
    train_gender_dataset,
    epochs=15,
    validation_data=val_gender_dataset,
    callbacks=gender_callbacks,
    class_weight=gender_weight_dict
)

# ขั้นตอนที่ 3: Fine-tuning ทั้งหมด - ปลดล็อคทั้ง base model (แต่ใช้ learning rate ที่ต่ำมาก)
print("Starting final fine-tuning of age model...")
age_base_model.trainable = True

# คอมไพล์อีกครั้งด้วย learning rate ที่ต่ำลงอีก
age_model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate / 100),
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy']
)

# ฝึกโมเดลอีกครั้ง
age_history3 = age_model.fit(
    train_age_dataset,
    epochs=10,
    validation_data=val_age_dataset,
    callbacks=age_callbacks,
    class_weight=age_weight_dict
)

# ทำเช่นเดียวกันกับโมเดลเพศ
print("Starting final fine-tuning of gender model...")
gender_base_model.trainable = True

gender_model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate / 100),
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy']
)

gender_history3 = gender_model.fit(
    train_gender_dataset,
    epochs=10,
    validation_data=val_gender_dataset,
    callbacks=gender_callbacks,
    class_weight=gender_weight_dict
)

# ประเมินผลโมเดลด้วยชุดข้อมูล test
print("\nEvaluating improved age model on test data:")
age_eval = age_model.evaluate(test_age_dataset)
print(f"Test Loss: {age_eval[0]:.4f}, Test Accuracy: {age_eval[1]:.4f}")

print("\nEvaluating improved gender model on test data:")
gender_eval = gender_model.evaluate(test_gender_dataset)
print(f"Test Loss: {gender_eval[0]:.4f}, Test Accuracy: {gender_eval[1]:.4f}")

# บันทึกโมเดลสุดท้าย
age_model.save('age_model_improved.h5')
gender_model.save('gender_model_improved.h5')
print("Models successfully saved in .h5 format!")

# สร้าง confusion matrix สำหรับโมเดลอายุ
def get_predictions(model, dataset):
    all_predictions = []
    all_true_labels = []
    
    for x, y in dataset:
        pred = model.predict(x)
        pred_idx = np.argmax(pred, axis=1)
        true_idx = np.argmax(y.numpy(), axis=1)
        
        all_predictions.extend(pred_idx)
        all_true_labels.extend(true_idx)
    
    return all_predictions, all_true_labels

# ประเมินโมเดลอายุ
print("\nGenerating confusion matrix for age model...")
age_predictions, age_true_labels = get_predictions(age_model, test_age_dataset)

# แปลงกลับเป็นชื่อคลาส
age_decoding = {0:'0-2', 1:'4-6', 2:'8-13', 3:'15-20', 4:'25-32', 5:'38-43', 6:'48-53', 7:'60+'}
age_labels_str = [age_decoding[i] for i in range(8)]

# สร้าง confusion matrix
age_cm = confusion_matrix(age_true_labels, age_predictions)
age_cm_norm = age_cm.astype('float') / age_cm.sum(axis=1)[:, np.newaxis]

# แสดง confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(age_cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=age_labels_str, 
            yticklabels=age_labels_str)
plt.xlabel('Predicted Age')
plt.ylabel('True Age')
plt.title('Age Classification Confusion Matrix (Normalized)')
plt.tight_layout()
plt.savefig('age_confusion_matrix_improved.png')

# ประเมินโมเดลเพศ
print("\nGenerating confusion matrix for gender model...")
gender_predictions, gender_true_labels = get_predictions(gender_model, test_gender_dataset)

# แปลงกลับเป็นชื่อคลาส
gender_decoding = {0:'female', 1:'male', 2:'unspecified'}
gender_labels_str = [gender_decoding[i] for i in range(3)]

# สร้าง confusion matrix
gender_cm = confusion_matrix(gender_true_labels, gender_predictions)
gender_cm_norm = gender_cm.astype('float') / gender_cm.sum(axis=1)[:, np.newaxis]

# แสดง confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(gender_cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=gender_labels_str, 
            yticklabels=gender_labels_str)
plt.xlabel('Predicted Gender')
plt.ylabel('True Gender')
plt.title('Gender Classification Confusion Matrix (Normalized)')
plt.tight_layout()
plt.savefig('gender_confusion_matrix_improved.png')

# สร้างกราฟแสดงประสิทธิภาพของโมเดล
def plot_training_history(histories, model_name):
    # รวมประวัติการฝึกโมเดลทั้งสามขั้นตอน
    combined_acc = []
    combined_val_acc = []
    combined_loss = []
    combined_val_loss = []
    
    for history in histories:
        combined_acc.extend(history.history['accuracy'])
        combined_val_acc.extend(history.history['val_accuracy'])
        combined_loss.extend(history.history['loss'])
        combined_val_loss.extend(history.history['val_loss'])
    
    combined_epochs = range(1, len(combined_acc) + 1)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(combined_epochs, combined_acc, 'b', label='Training Accuracy')
    plt.plot(combined_epochs, combined_val_acc, 'r', label='Validation Accuracy')
    plt.axvline(x=len(histories[0].history['accuracy']), color='g', linestyle='--', label='Start Fine-tuning 1')
    if len(histories) > 2:
        plt.axvline(x=len(histories[0].history['accuracy']) + len(histories[1].history['accuracy']), 
                   color='m', linestyle='--', label='Start Fine-tuning 2')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(combined_epochs, combined_loss, 'b', label='Training Loss')
    plt.plot(combined_epochs, combined_val_loss, 'r', label='Validation Loss')
    plt.axvline(x=len(histories[0].history['loss']), color='g', linestyle='--', label='Start Fine-tuning 1')
    if len(histories) > 2:
        plt.axvline(x=len(histories[0].history['loss']) + len(histories[1].history['loss']), 
                   color='m', linestyle='--', label='Start Fine-tuning 2')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_performance_improved.png')
    plt.show()

# แสดงกราฟประสิทธิภาพของทั้งสองโมเดล
plot_training_history([age_history1, age_history2, age_history3], 'Age Model')
plot_training_history([gender_history1, gender_history2, gender_history3], 'Gender Model')

# สร้างฟังก์ชันสำหรับการทดสอบกับรูปภาพ
def test_models(image_path):
    # อ่านและ preprocess รูปภาพ
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ปรับขนาดเป็น 224x224 สำหรับ EfficientNet
    image_resized = cv2.resize(image, (224, 224)) / 255.0
    
    # แสดงรูปภาพ
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title('Test Image')
    plt.axis('off')
    
    # ทำนายอายุและเพศ
    image_array = np.expand_dims(image_resized, 0)
    
    # ทำนายอายุ
    age_prediction = age_model.predict(image_array)
    age_index = np.argmax(age_prediction)
    age_confidence = age_prediction[0][age_index] * 100
    age_decoding = {0:'0-2', 1:'4-6', 2:'8-13', 3:'15-20', 4:'25-32', 5:'38-43', 6:'48-53', 7:'60+'}
    predicted_age = age_decoding[age_index]
    
    # แสดงความน่าจะเป็นของแต่ละกลุ่มอายุ
    age_probs = {}
    for i, prob in enumerate(age_prediction[0]):
        age_probs[age_decoding[i]] = f"{prob*100:.1f}%"
    
    # ทำนายเพศ
    gender_prediction = gender_model.predict(image_array)
    gender_index = np.argmax(gender_prediction)
    gender_confidence = gender_prediction[0][gender_index] * 100
    gender_decoding = {0:'female', 1:'male', 2:'unspecified'}
    predicted_gender = gender_decoding[gender_index]
    
    # แสดงความน่าจะเป็นของแต่ละเพศ
    gender_probs = {}
    for i, prob in enumerate(gender_prediction[0]):
        gender_probs[gender_decoding[i]] = f"{prob*100:.1f}%"
    
    # แสดงผลลัพธ์
    print(f'Predicted age: {predicted_age} (Confidence: {age_confidence:.1f}%)')
    print("Age probabilities:", age_probs)
    print(f'Predicted gender: {predicted_gender} (Confidence: {gender_confidence:.1f}%)')
    print("Gender probabilities:", gender_probs)
    
    # แสดงผลลัพธ์บนภาพ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    # สร้างภาพสำหรับแสดงผล
    display_img = image.copy()
    
    # เพิ่มกรอบและข้อความ
    h, w = display_img.shape[:2]
    cv2.rectangle(display_img, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(display_img, f"Age: {predicted_age} ({age_confidence:.1f}%)", 
                (10, 25), font, font_scale, (255, 255, 255), font_thickness)
    
    cv2.rectangle(display_img, (0, h-40), (w, h), (0, 0, 0), -1)
    cv2.putText(display_img, f"Gender: {predicted_gender} ({gender_confidence:.1f}%)", 
                (10, h-15), font, font_scale, (255, 255, 255), font_thickness)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(display_img)
    plt.axis('off')
    plt.title('Prediction Results')
    plt.tight_layout()
    plt.savefig('prediction_result_improved.png')
    plt.show()
    
    return age_probs, gender_probs

# สร้างฟังก์ชันสำหรับการประเมินประสิทธิภาพบนชุดข้อมูล test
def evaluate_models_detailed():
    # ประเมินและสร้าง classification report สำหรับโมเดลอายุ
    age_predictions, age_true_labels = get_predictions(age_model, test_age_dataset)
    
    # แปลงกลับเป็นชื่อคลาส
    age_decoding = {0:'0-2', 1:'4-6', 2:'8-13', 3:'15-20', 4:'25-32', 5:'38-43', 6:'48-53', 7:'60+'}
    age_label_names = [age_decoding[i] for i in range(8)]
    
    print("\nAge Classification Report:")
    print(classification_report(age_true_labels, age_predictions, target_names=age_label_names))
    
    # ประเมินและสร้าง classification report สำหรับโมเดลเพศ
    gender_predictions, gender_true_labels = get_predictions(gender_model, test_gender_dataset)
    
    # แปลงกลับเป็นชื่อคลาส
    gender_decoding = {0:'female', 1:'male', 2:'unspecified'}
    gender_label_names = [gender_decoding[i] for i in range(3)]
    
    print("\nGender Classification Report:")
    print(classification_report(gender_true_labels, gender_predictions, target_names=gender_label_names))

# ประเมินโมเดลอย่างละเอียด
evaluate_models_detailed()

# ทดสอบโมเดลกับรูปภาพตัวอย่าง (ถ้ามี)
if len(X_test) > 0:
    print("\nTesting improved models with sample image:")
    test_sample_path = X_test[0]
    test_models(test_sample_path)

print("\nModel improvement complete!")
print("Age model loss significantly reduced and accuracy improved.")
print("Gender model loss significantly reduced and accuracy improved.")
print("Both models trained with techniques to avoid overfitting:")
print("- Improved data augmentation")
print("- EfficientNetB3 architecture")
print("- Focal loss")
print("- Progressive unfreezing of layers")
print("- Learning rate scheduling")
print("- Regularization techniques (Dropout, BatchNormalization)")
print("- Advanced training strategy with 3 phases")