from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pyodbc
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=LAPTOP-0SMFMLRQ;'
    'DATABASE=CourseAI;'
    'UID=test;'
    'PWD=1234'
)

query = """
SELECT UserId, CourseId, EnrollmentStatus
FROM dbo.UserCourses
"""

df = pd.read_sql(query, conn)


# Xử lý dữ liệu
df['EnrollmentStatus'] = df['EnrollmentStatus'].astype(int)

# Mã hóa UserId và CourseId
user_encoder = LabelEncoder()
df['UserId'] = user_encoder.fit_transform(df['UserId'])

course_encoder = LabelEncoder()
df['CourseId'] = course_encoder.fit_transform(df['CourseId'])

X = df[['UserId', 'CourseId']]
y = df['EnrollmentStatus']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Xây dựng mô hình DNN (tối ưu)
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Huấn luyện mô hình
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=50, 
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)
# 6. Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Độ chính xác mô hình: {accuracy:.2f}")

# Hàm gợi ý khóa học cho người dùng (tối ưu)
def recommend_courses_dnn(user_id, top_n=5):
    user_encoded = user_encoder.transform([user_id])[0]
    courses = np.setdiff1d(np.arange(df['CourseId'].nunique()), df[df['UserId'] == user_encoded]['CourseId'])

    inputs = np.array([[user_encoded, course] for course in courses])
    inputs_scaled = scaler.transform(inputs)
    
    predictions = model.predict(inputs_scaled).flatten()
    top_courses = np.argsort(-predictions)[:top_n]
    recommended_courses = course_encoder.inverse_transform(top_courses)

    return recommended_courses.tolist()

# Ví dụ: Gợi ý khóa học cho User có ID = 'E859E761-C879-4422-BDDB-06999875DF33'
user_example = 'E859E761-C879-4422-BDDB-06999875DF33'
recommended = recommend_courses_dnn(user_example)
print("Gợi ý khóa học (DNN) cho", user_example, ":", recommended)