# frontend/Dockerfile

FROM python:3.10.6

WORKDIR /APP

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
