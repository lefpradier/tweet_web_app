import requests
import pandas as pd
import numpy as np

test_df = pd.read_csv("data/raw/raw_test.csv")
host = "localhost"
port = "5000"
url = f"http://{host}:{port}/invocations"
headers = {"Content-Type": "application/json"}
test_df["preds"] = np.nan
for index, row in test_df.iterrows():
    r = requests.post(url=url, headers=headers, data={"data": row["text"]})
    test_df.loc[index, "preds"] = r.text
test_df.to_csv("test_predictions", index=False)
