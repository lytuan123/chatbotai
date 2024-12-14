FROM python:3.9-slim

WORKDIR /app

# Cài đặt các phụ thuộc
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

