# backend/Dockerfile

FROM python:3.10.6

WORKDIR /APP

COPY requirements.txt .
COPY model/requirements.txt model_requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r model_requirements.txt
RUN python -m spacy download "en_core_web_sm"

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
