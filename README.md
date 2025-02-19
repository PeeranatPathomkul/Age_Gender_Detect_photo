# Age_Gender_Detect_photo
6610110214 Peeranat Pathomkul
# Gender Classification Model

โปรเจกต์นี้เป็นระบบจำแนกเพศจากรูปภาพโดยใช้ Deep Learning (Convolutional Neural Network) ด้วย TensorFlow/Keras

## โครงสร้างโปรเจกต์
```
project/
│
├── Training/          # ข้อมูลสำหรับฝึกฝนโมเดล
│   ├── female/       # รูปภาพผู้หญิงสำหรับฝึกฝน
│   └── male/         # รูปภาพผู้ชายสำหรับฝึกฝน
│
├── Validation/       # ข้อมูลสำหรับทดสอบโมเดล
│   ├── female/      # รูปภาพผู้หญิงสำหรับทดสอบ
│   └── male/        # รูปภาพผู้ชายสำหรับทดสอบ
│
├── model_gen.py     # โค้ดหลักสำหรับสร้างและฝึกฝนโมเดล
├── main.ipynb       # Jupyter notebook สำหรับการทดลอง
├── model_age.ipynb  # Notebook สำหรับการวิเคราะห์อายุ
└── model_gender.ipynb  # Notebook สำหรับการวิเคราะห์เพศ
```

## คุณสมบัติ
- จำแนกรูปภาพเป็นเพศชายหรือหญิง
- ใช้ CNN architecture ที่มีประสิทธิภาพ
- รองรับการ Data Augmentation เพื่อเพิ่มประสิทธิภาพ
- มีระบบ Early Stopping และ Model Checkpoint
- รองรับการทำนายรูปภาพใหม่

## ข้อกำหนดระบบ
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pillow (PIL)

## การติดตั้ง
1. โคลนโปรเจกต์:
```bash
git clone [URL ของ repository]
```

2. ติดตั้ง dependencies:
```bash
pip install tensorflow numpy pillow
```

## การใช้งาน
1. จัดเตรียมข้อมูล:
   - วางรูปภาพผู้ชายในโฟลเดอร์ Training/male และ Validation/male
   - วางรูปภาพผู้หญิงในโฟลเดอร์ Training/female และ Validation/female

2. ฝึกฝนโมเดล:
```python
python model_gen.py
```

3. ทำนายรูปภาพใหม่:
```python
from model_gen import predict_gender
result = predict_gender(model, "path_to_image.jpg")
```

## การปรับแต่งพารามิเตอร์
สามารถปรับแต่งค่าต่างๆ ในไฟล์ model_gen.py:
- IMG_HEIGHT และ IMG_WIDTH: ขนาดรูปภาพ (default: 150x150)
- BATCH_SIZE: ขนาด batch (default: 32)
- EPOCHS: จำนวนรอบการฝึกฝน (default: 10)

## โครงสร้างโมเดล
- 3 Convolutional blocks พร้อม BatchNormalization
- Data augmentation (horizontal flip, rotation, zoom)
- Dropout layer เพื่อป้องกัน overfitting
- Binary classification ด้วย sigmoid activation

## ผู้พัฒนา
[ใส่ชื่อและข้อมูลติดต่อของคุณ]

## License
[ระบุ License ที่ใช้]
