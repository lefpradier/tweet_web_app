# 1. Serve the best model with MLFlow
# a. Create model for deployment
cd ..
python3 src/modeling/serving_model_input.py
# b. Create a REST API
mlflow models serve -m deployment/backend/model > /dev/null 2>&1 &
pid=$!
sleep 5
# c. Send test data to the API
python3 deployment/backend/test_serving.py
kill $pid
#2. Convert TensorFlow model to TFLite
python3 deployment/backend/convert_model.py
#3. Perform unit tests with pytest on the FastAPI API
cd deployment
python3 -m pytest backend/test_MAIN.py
#4. build local API and test with simple tweet
docker build -f backend/Dockerfile -t backend .
docker run -p 8080:8080 backend > /dev/null 2>&1 &
pid=$!
sleep 5
until curl -X 'POST' \
  'http://0.0.0.0:8080/tweet?tweet=i%20am%20sad' \
  -H 'accept: application/json' \
  -d ''; do sleep 5; done
kill $pid

#5. Commit and push the API to GitHub, from GitHub to Azure App Service
echo
echo
echo
echo "##############################################"
echo "You can now commit and push the API to GitHub."
echo "##############################################"
