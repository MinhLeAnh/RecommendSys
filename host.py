from fastapi import FastAPI, HTTPException
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import uvicorn
import pyodbc
from nltk.tokenize import word_tokenize
import nltk
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.model_selection import train_test_split
import itertools

app = FastAPI()

import os
from datetime import datetime

def is_data_updated():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=LAPTOP-0SMFMLRQ;'
        'DATABASE=CourseAI;'
        'UID=test;'
        'PWD=1234'
    )

    query = """
    SELECT 
        (SELECT COUNT(*) FROM Courses) AS CourseCount,
        (SELECT COUNT(*) FROM UserCourses) AS UserCourseCount
    """
    counts = pd.read_sql(query, conn)
    conn.close()
    
    # Đọc số lượng bản ghi trước đó
    if os.path.exists("last_data_count.txt"):
        with open("last_data_count.txt", "r") as f:
            last_count = f.read().strip().split(',')
            last_course_count = int(last_count[0])
            last_usercourse_count = int(last_count[1])
    else:
        last_course_count = last_usercourse_count = 0

    # Lấy số lượng bản ghi hiện tại
    current_course_count = counts.iloc[0]['CourseCount']
    current_usercourse_count = counts.iloc[0]['UserCourseCount']

    # Cập nhật số lượng bản ghi vào file
    with open("last_data_count.txt", "w") as f:
        f.write(f"{current_course_count},{current_usercourse_count}")

    # Kiểm tra nếu có thay đổi
    return (current_course_count != last_course_count) or (current_usercourse_count != last_usercourse_count)

#_______________________


# Load Word2Vec model
word2vec_model_path = "saved_models/word2vec_model.model"
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)

# Load DNN model
dnn_model_path = "saved_models/dnn_model.h5"
dnn_model = tf.keras.models.load_model(dnn_model_path)
scaler = StandardScaler()

# Kết nối tới SQL Server
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=LAPTOP-0SMFMLRQ;'
    'DATABASE=CourseAI;'
    'UID=test;'
    'PWD=1234'
)

# Truy vấn dữ liệu từ bảng Courses với các trường đầy đủ
query_courses = """
SELECT 
    Id, Title, Description, Image, Price, InstructorInfo, Level, Duration
FROM Courses
"""
df_courses = pd.read_sql(query_courses, conn)
df_courses.fillna('', inplace=True)

# Tạo cột combined_text nếu chưa có
df_courses['combined_text'] = (
    df_courses['Title'] + ' ' + df_courses['Description']
)

# Tạo embedding cho mỗi khóa học
def get_text_embedding(text_tokens):
    vectors = [word2vec_model.wv[word] for word in text_tokens if word in word2vec_model.wv]
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(word2vec_model.vector_size)

# Tạo tokenized text cho từng khóa học
df_courses['tokenized_text'] = df_courses['combined_text'].apply(lambda x: word_tokenize(x.lower()))
df_courses['embedding'] = df_courses['tokenized_text'].apply(get_text_embedding)

# Truy vấn dữ liệu từ bảng UserCourses
query_users = """
SELECT UserId, CourseId, EnrollmentStatus
FROM dbo.UserCourses
"""
df_users = pd.read_sql(query_users, conn)

# Đóng kết nối
conn.close()

# Label Encoding for Users and Courses
user_encoder = LabelEncoder()
course_encoder = LabelEncoder()
df_users['UserId'] = user_encoder.fit_transform(df_users['UserId'])
df_users['CourseId'] = course_encoder.fit_transform(df_users['CourseId'])

# train lại nếu cần
def train_models_if_needed():
    if is_data_updated():
        print("Training model with new data...")
        nltk.download('punkt')

        # Kết nối SQL Server
        conn = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=LAPTOP-0SMFMLRQ;'
            'DATABASE=CourseAI;'
        )

        # Truy vấn dữ liệu từ bảng Courses
        query_courses = """
        SELECT Id, Title, Description
        FROM Courses
        """
        df_courses = pd.read_sql(query_courses, conn)

        query = """
        SELECT UserId, CourseId, EnrollmentStatus
        FROM dbo.UserCourses
        """

        df = pd.read_sql(query, conn)

        conn.close()

        # Xử lý dữ liệu văn bản khóa học
        df_courses.fillna('', inplace=True)
        df_courses['combined_text'] = (
            df_courses['Title'] + ' ' + df_courses['Description']
        )

        # Tách từ trong văn bản
        def tokenize_text(text):
            return word_tokenize(text.lower())

        df_courses['tokenized_text'] = df_courses['combined_text'].apply(tokenize_text)

        # Huấn luyện mô hình Word2Vec trên các văn bản khóa học
        word2vec_model = gensim.models.Word2Vec(
            sentences=df_courses['tokenized_text'].tolist(), 
            vector_size=100,  # Kích thước vector
            window=5,         # Kích thước ngữ cảnh
            min_count=1,      # Số lần xuất hiện tối thiểu của từ
            workers=4
        )

        #  Truy vấn dữ liệu người dùng và khóa học
        query = "SELECT UserId, CourseId, EnrollmentStatus FROM dbo.UserCourses"
        df_enrollments = pd.read_sql(query, conn)
        conn.close()

        # Đặt EnrollmentStatus cho khóa học chưa đăng ký là 0
        df_enrollments['EnrollmentStatus'] = df_enrollments['EnrollmentStatus'].fillna(0).apply(lambda x: 1 if x == 2 else 0)

        # Mã hóa UserId và CourseId
        user_encoder = LabelEncoder()
        course_encoder = LabelEncoder()

        df_enrollments['UserId'] = user_encoder.fit_transform(df_enrollments['UserId'])
        df_enrollments['CourseId'] = course_encoder.fit_transform(df_enrollments['CourseId'])

        # Tạo tất cả kết hợp UserId - CourseId (bao gồm cả các khóa học chưa đăng ký)
        all_users = df_enrollments['UserId'].unique()
        all_courses = df_enrollments['CourseId'].unique()
        all_combinations = pd.DataFrame(list(itertools.product(all_users, all_courses)), columns=['UserId', 'CourseId'])

        # Gán trạng thái đăng ký cho các khóa học (0 cho khóa học chưa đăng ký)
        df = pd.merge(all_combinations, df_enrollments[['UserId', 'CourseId', 'EnrollmentStatus']],
                    on=['UserId', 'CourseId'], how='left')
        df['EnrollmentStatus'] = df['EnrollmentStatus'].fillna(0)

        # Định dạng lại cột dữ liệu
        X = df[['UserId', 'CourseId']]
        y = df['EnrollmentStatus']

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


        # Tạo ma trận UserId x CourseId
        df_pivot = df.pivot(index='UserId', columns='CourseId', values='EnrollmentStatus')

        # Xuất ma trận ra file CSV
        df_pivot.to_csv('user_course_matrix.csv', index=True)

        # Xây dựng mô hình DNN
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
        #lưu model__________________________________
        # Đường dẫn lưu model
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)

        # Lưu mô hình Word2Vec
        word2vec_model_path = os.path.join(model_dir, "word2vec_model.model")
        word2vec_model.save(word2vec_model_path)
        print(f"Word2Vec model saved at: {word2vec_model_path}")

        # Lưu mô hình DNN
        dnn_model_path = os.path.join(model_dir, "dnn_model.h5")
        model.save(dnn_model_path)
        print(f"DNN model saved at: {dnn_model_path}")

        with open("last_training.txt", "w") as f:
            f.write(datetime.now().isoformat())
        print("Model training completed.")
    else:
        print("No new data. Model training skipped.")

@app.get("/recommend/similar")
def recommend_similar_courses(course_id: str, top_n: int = 5):
    course_id = course_id.upper()
    course = df_courses[df_courses['Id'] == course_id]
    if course.empty:
        raise HTTPException(status_code=404, detail="Khóa học không tồn tại.")
    
    # Lấy embedding của khóa học
    course_embedding = course.iloc[0]['embedding']

    # Lấy tất cả embedding của các khóa học
    all_embeddings = np.stack(df_courses['embedding'].values)

    # Tính cosine similarity
    similarities = cosine_similarity([course_embedding], all_embeddings).flatten()
    similarities[course.index[0]] = -1  # Loại bỏ chính khóa học

    # Đề xuất khóa học
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommended_courses = df_courses.iloc[top_indices][['Id', 'Title', 'Description', 'Image', 'Price', 'InstructorInfo', 'Level', 'Duration']].to_dict(orient='records')

    return {"recommended_courses": recommended_courses}

@app.get("/recommend/personal")
def recommend_personal_courses(user_id: str, top_n: int = 5):
    user_id = user_id.upper()
    if user_id not in user_encoder.classes_:
        train_models_if_needed()
    user_encoded = user_encoder.transform([user_id])[0]

    courses = np.setdiff1d(np.arange(df_users['CourseId'].nunique()), df_users[df_users['UserId'] == user_encoded]['CourseId'])

    inputs = np.array([[user_encoded, course] for course in courses])
    inputs_scaled = scaler.fit_transform(inputs)

    predictions = dnn_model.predict(inputs_scaled).flatten()
    top_courses = np.argsort(-predictions)[:top_n]
    recommended_courses = course_encoder.inverse_transform(top_courses).tolist()

    # Lấy thông tin khóa học theo ID
    recommended_courses_details = df_courses[df_courses['Id'].isin(recommended_courses)][['Id', 'Title', 'Description', 'Image', 'Price', 'InstructorInfo', 'Level', 'Duration']].to_dict(orient='records')

    return {"recommended_courses": recommended_courses_details}

@app.on_event("startup")
def on_startup():
    train_models_if_needed()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
