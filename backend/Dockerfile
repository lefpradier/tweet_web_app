# backend/Dockerfile

FROM python:3.10.6

WORKDIR /APP

COPY backend/requirements.txt .

RUN pip install -r requirements.txt

RUN python -m spacy download "en_core_web_sm"

COPY . .

EXPOSE 8080

CMD ["python", "backend/main.py"]
