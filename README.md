## Cách chạy:  
Bước 1: Vào thư mục dự án  
Bước 2: Tải các thư viện cần thiết
`pip install requirement.txt`   
Bước 3: Sửa các chuỗi kết nối trong file host.py  
Bước 4: Chạy lệnh trong terminal  
`uvicorn host:app --host 127.0.0.1 --port 8000 --reload`  
Bước 5: Vào trang http://localhost:8000/docs#/  (lưu ý nếu không load được thì vào lại vs code, vào file host.py và bấm ctrl S là được)
