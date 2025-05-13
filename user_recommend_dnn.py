from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pyodbc
import itertools

# 🚀 Kết nối cơ sở dữ liệu
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=LAPTOP-0SMFMLRQ;'
    'DATABASE=CourseAI;'
    'UID=test;'
    'PWD=1234'
)

# 🚀 Truy vấn dữ liệu người dùng và khóa học
query = "SELECT UserId, CourseId, EnrollmentStatus FROM dbo.UserCourses"
df_enrollments = pd.read_sql(query, conn)
conn.close()

# 🚀 Đặt EnrollmentStatus cho khóa học chưa đăng ký là 0
df_enrollments['EnrollmentStatus'] = df_enrollments['EnrollmentStatus'].fillna(0).apply(lambda x: 1 if x == 2 else 0)

# 🚀 Mã hóa UserId và CourseId
user_encoder = LabelEncoder()
course_encoder = LabelEncoder()

df_enrollments['UserId'] = user_encoder.fit_transform(df_enrollments['UserId'])
df_enrollments['CourseId'] = course_encoder.fit_transform(df_enrollments['CourseId'])

# 🚀 Tạo tất cả kết hợp UserId - CourseId (bao gồm cả các khóa học chưa đăng ký)
all_users = df_enrollments['UserId'].unique()
all_courses = df_enrollments['CourseId'].unique()
all_combinations = pd.DataFrame(list(itertools.product(all_users, all_courses)), columns=['UserId', 'CourseId'])

# 🚀 Gán trạng thái đăng ký cho các khóa học (0 cho khóa học chưa đăng ký)
df = pd.merge(all_combinations, df_enrollments[['UserId', 'CourseId', 'EnrollmentStatus']],
              on=['UserId', 'CourseId'], how='left')
df['EnrollmentStatus'] = df['EnrollmentStatus'].fillna(0)

# 🚀 Định dạng lại cột dữ liệu
X = df[['UserId', 'CourseId']]
y = df['EnrollmentStatus']

# 🚀 Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🚀 Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Tạo ma trận UserId x CourseId
df_pivot = df.pivot(index='UserId', columns='CourseId', values='EnrollmentStatus')

# Xuất ma trận ra file CSV
df_pivot.to_csv('user_course_matrix.csv', index=True)

# 🚀 Xây dựng mô hình DNN (tối ưu)
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

# 🚀 Huấn luyện mô hình
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

# 🚀 Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Độ chính xác mô hình: {accuracy:.2f}")

# 🚀 Hàm gợi ý khóa học cho người dùng
def recommend_courses_dnn(user_id, top_n=5):
    # Kiểm tra nếu user_id không tồn tại
    if user_id not in user_encoder.classes_:
        raise ValueError("UserId không tồn tại trong dữ liệu.")

    # Mã hóa user_id
    user_encoded = user_encoder.transform([user_id])[0]

    # 🚀 Lấy khóa học mà người dùng đã thực sự đăng ký
    courses_user_registered = df[(df['UserId'] == user_encoded) & (df['EnrollmentStatus'] == 1)]['CourseId'].unique()
    all_courses_encoded = df['CourseId'].unique()

    # Khóa học chưa đăng ký
    courses_not_registered = np.setdiff1d(all_courses_encoded, courses_user_registered)

    # Kiểm tra nếu không còn khóa học nào để gợi ý
    if len(courses_not_registered) == 0:
        return "Người dùng đã đăng ký tất cả các khóa học. Không còn khóa học nào để gợi ý."

    # Dự đoán cho khóa học chưa đăng ký
    inputs = np.array([[user_encoded, course] for course in courses_not_registered])
    inputs_scaled = scaler.transform(inputs)
    predictions = model.predict(inputs_scaled).flatten()

    # Lấy top N khóa học gợi ý
    top_courses_indices = np.argsort(-predictions)[:top_n]
    top_courses = courses_not_registered[top_courses_indices]
    recommended_courses = course_encoder.inverse_transform(top_courses)

    return recommended_courses.tolist()

# 🚀 Ví dụ: Gợi ý khóa học cho User có ID = 'E859E761-C879-4422-BDDB-06999875DF33'
user_example = 'E859E761-C879-4422-BDDB-06999875DF33'

# 🚀 Kiểm tra UserId
try:
    recommended = recommend_courses_dnn(user_example)
    print("Gợi ý khóa học (DNN) cho", user_example, ":", recommended)
except ValueError as e:
    print(e)

