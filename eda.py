import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import itertools
import pyodbc
# Kết nối với SQL Server
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=LAPTOP-0SMFMLRQ;'
    'DATABASE=CourseAI;'
    'UID=test;'
    'PWD=1234'
)

# Truy vấn dữ liệu người dùng và khóa học
query_users = "SELECT DISTINCT UserId FROM dbo.UserCourses"
query_courses = "SELECT DISTINCT CourseId FROM dbo.UserCourses"
query_enrollment = "SELECT UserId, CourseId, EnrollmentStatus FROM dbo.UserCourses"

df_users = pd.read_sql(query_users, conn)
df_courses = pd.read_sql(query_courses, conn)
df_enrollments = pd.read_sql(query_enrollment, conn)
conn.close()

# Tạo tất cả các kết hợp giữa người dùng và khóa học
all_combinations = pd.DataFrame(list(itertools.product(df_users['UserId'], df_courses['CourseId'])), 
                                columns=['UserId', 'CourseId'])

# Kết hợp với bảng UserCourses để xác định khóa học đã đăng ký
df = pd.merge(all_combinations, df_enrollments, on=['UserId', 'CourseId'], how='left')

# Đặt EnrollmentStatus cho khóa học chưa đăng ký là 0
df['EnrollmentStatus'] = df['EnrollmentStatus'].fillna(0).apply(lambda x: 1 if x == 2 else 0)

# Mã hóa UserId và CourseId
user_encoder = LabelEncoder()
df['UserId'] = user_encoder.fit_transform(df['UserId'])

course_encoder = LabelEncoder()
df['CourseId'] = course_encoder.fit_transform(df['CourseId'])

# Định dạng lại cột dữ liệu
X = df[['UserId', 'CourseId']]
y = df['EnrollmentStatus']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def evaluate_input_data(df):
    print("\n🔎 Đánh giá dữ liệu đầu vào:")
    
    # 1. Kiểm tra dữ liệu thiếu (Missing Values)
    missing_values = df.isnull().sum()
    print("\n1️⃣ Dữ liệu thiếu (Missing Values):")
    print(missing_values[missing_values > 0])
    
    # 2. Kiểm tra phân phối của các giá trị UserId và CourseId
    print("\n2️⃣ Phân phối UserId và CourseId (sau khi mã hóa):")
    print("Số lượng UserId duy nhất:", df['UserId'].nunique())
    print("Số lượng CourseId duy nhất:", df['CourseId'].nunique())
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['UserId'], bins=30, color='skyblue')
    plt.title("Phân phối UserId")

    plt.subplot(1, 2, 2)
    sns.histplot(df['CourseId'], bins=30, color='lightgreen')
    plt.title("Phân phối CourseId")
    plt.show()

    # 3. Kiểm tra phân phối dữ liệu sau khi chuẩn hóa
    print("\n3️⃣ Phân phối dữ liệu sau khi chuẩn hóa:")
    X_scaled_df = pd.DataFrame(X_scaled, columns=['UserId_Scaled', 'CourseId_Scaled'])
    plt.figure(figsize=(12, 5))
    sns.histplot(X_scaled_df['UserId_Scaled'], color='skyblue', label='UserId_Scaled')
    sns.histplot(X_scaled_df['CourseId_Scaled'], color='lightgreen', label='CourseId_Scaled')
    plt.title("Phân phối UserId và CourseId sau khi chuẩn hóa")
    plt.legend()
    plt.show()
    
    # 4. Kiểm tra giá trị không hợp lệ trong EnrollmentStatus
    print("\n4️⃣ Giá trị không hợp lệ trong EnrollmentStatus:")
    invalid_status = df[~df['EnrollmentStatus'].isin([0, 1])]
    if not invalid_status.empty:
        print("Tìm thấy giá trị không hợp lệ:")
        print(invalid_status)
    else:
        print("Không có giá trị không hợp lệ.")
    
    # 5. Phân tích tỷ lệ lớp trong EnrollmentStatus
    print("\n5️⃣ Phân tích tỷ lệ lớp (EnrollmentStatus):")
    print(df['EnrollmentStatus'].value_counts(normalize=True))
    sns.countplot(x=df['EnrollmentStatus'], palette='pastel')
    plt.title("Phân bố của EnrollmentStatus")
    plt.show()

# Gọi hàm đánh giá dữ liệu đầu vào
evaluate_input_data(df)
