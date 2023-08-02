#1. serve MM
#a.Create model for deployment
python serving_model_input.py
#2. Use test dataset for serving test
#a.create restapi
mlflow models serve -m deployment/backend/model > /dev/null 2>&1 &
pid=$!
sleep 5
#b.send data to api
python test_serving.py
kill $pid
#3. convert to TFLite
python convert_model.py
#4. unit tests
python -m pytest backend/test_MAIN.py
#5. build local API and test with simple tweet
docker build -f backend/Dockerfile -t backend .
docker run -p 8080:8080 backend > /dev/null 2>&1 &
pid=$!
sleep 5
until curl -X 'POST' \
  'http://0.0.0.0:8080/tweet?tweet=i%20am%20sad' \
  -H 'accept: application/json' \
  -d ''; do sleep 5; done
kill $pid

#6. azure deployment : commit and push API to git, from git to azure

#todo:tests unitaires :
#*ping
#*good item
#!bad items
#todo:sanity check models and api
#todo:print and comments along pipe
#todo:check path
#todo:integrate stage commit push to main pipe
#todo:architecture init