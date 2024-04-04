# [Tweet web app](https://lefpradier.github.io/tweet_web_app)
Ce repository contient le code source d'une API permettant de prédire le sentiment majoritaire d'un tweet, et d'une application Streamlit faisant appel à cette API. L'API et l'application ont été initialement déployées aux adresses suivantes : https://sentiment-analysis-lefp.azurewebsites.net et https://tweet-sentiment-lefp.streamlit.app/.
## Installation locale
### API
Cette API peut être installée comme une image Docker.
````
docker build . -t backend -f backend/Dockerfile # compile l'image
docker run -p 8080:8080 backend # exécute l'image Docker
````

### Application
Pour exécuter l'application, il est nécessaire d'installer ses dépendances.
````
pip install -r requirements.txt
````
Elle peut ensuite être exécutée ainsi :
````
streamlit frontend_main.py
````

## Caractéristiques du modèle
Le modèle exposé par l'API prédit la probabilité qu'un tweet en anglais exprime un sentiment globalement négatif : 1 si le tweet est purement négatif, 0 s'il est purement positif. Ce modèle est une version compressée par TFLite d'un réseau de neurones plus complexe. Ce dernier possède une couche d'embedding <i>fasttext-commoncrawl</i> et une couche LSTM.<br>
Les entrées sont limitées à des séquences de 30 mots au maximum. Il est possible que certains mots ne soient pas reconnus car ils étaient trop rares dans le corpus original <i>Sentiment140</i>.
