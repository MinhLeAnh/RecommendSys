# Hướng dẫn cài đặt và chạy dự án: Hệ thống gợi ý (Recommendation System) được xây dựng bằng Python và FastAPI
## Bước 1: 
Kéo dự án về máy local dùng câu lệnh: git clone https://github.com/MinhLeAnh/RecommendSys.git
## Bước 2: Tải các thư viện cần thiết
`pip install requirement.txt`   
## Bước 3: 
Sửa các chuỗi kết nối trong file host.py  
## Bước 4: Chạy lệnh trong terminal  
`uvicorn host:app --host 127.0.0.1 --port 8000 --reload`  
## Bước 5: Vào trang http://localhost:8000/docs#/  
(lưu ý nếu không load được thì vào lại vs code, vào file host.py và bấm ctrl S là được)
