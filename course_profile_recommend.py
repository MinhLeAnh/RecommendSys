import pandas as pd
import pyodbc
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

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

# Hàm lấy vector của văn bản dựa trên Word2Vec
def get_text_embedding(text_tokens, model):
    vectors = [model.wv[word] for word in text_tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Tính vector cho khóa học
df_courses['embedding'] = df_courses['tokenized_text'].apply(lambda x: get_text_embedding(x, word2vec_model))

# Hàm trả về top 5 khóa học tương tự cho một khóa học
def get_top_5_similar_courses(course_id):
    course_row = df_courses[df_courses['Id'] == course_id]
    if course_row.empty:
        return "Khóa học không tồn tại."
    
    course_embedding = course_row.iloc[0]['embedding']
    all_embeddings = np.stack(df_courses['embedding'].to_numpy())

    # Tính độ tương đồng cosine giữa khóa học được chọn và tất cả khóa học khác
    similarities = cosine_similarity([course_embedding], all_embeddings).flatten()
    similarities[course_row.index[0]] = -1  # Loại bỏ chính khóa học đó

    # Lấy top 5 khóa học tương tự
    top_indices = similarities.argsort()[-5:][::-1]
    recommended_courses = df_courses.iloc[top_indices][['Id', 'Title', 'Description']]
    
    return recommended_courses

# Ví dụ: Gợi ý các khóa học tương tự cho khóa học có ID = 'C1'
course_id = 'FA190759-BD43-4584-BEDD-09180976641B'
recommended_courses = get_top_5_similar_courses(course_id)
print("Top 5 khóa học tương tự cho khóa học:", course_id)
print(recommended_courses)