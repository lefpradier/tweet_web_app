import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

##! STREAMLITE


# # https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
# st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Tweet sentiment analysis web app")

# displays a file uploader widget
tweet = st.text_input("Import a tweet")

# displays a button
if st.button("Predict sentiment"):
    if tweet is not None:
        data = {"tweet": tweet}
        res = requests.post(
            "https://sentiment-analysis-lefp.azurewebsites.net/tweet", params=data
        )
        output = res.json()
        if "score" in output:
            score = output.get("score")
            if score >= 0.5:
                pos_txt = score / 2
                score_txt = "Negativity of " + str(round(score * 100, 1)) + "%"
                image = plt.imread("frontend/sad.png")
            else:
                pos_txt = (1 - score) / 2 + score
                score_txt = "Positivity of " + str(round((1 - score) * 100, 1)) + "%"
                image = plt.imread("frontend/happy.png")

            fig = plt.figure(figsize=(5, 2), facecolor="#0E1117")
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1])
            axs = {}

            axs["img"] = fig.add_subplot(gs[0, :])
            im = axs["img"].imshow(image)

            axs["bar"] = fig.add_subplot(gs[1, :])
            axs["bar"].barh(0, score, color="red")
            axs["bar"].barh(0, (1 - score), color="green", left=score)
            axs["bar"].text(
                pos_txt,
                0,
                score_txt,
                color="white",
                fontsize=11,
                va="center",
                ha="center",
            )
            axs["bar"].set_facecolor("#0E1117")
            axs["bar"].set_axis_off()
            axs["img"].set_facecolor("#0E1117")
            axs["img"].set_axis_off()
            st.pyplot(fig)
        else:
            st.text("Error detail:")
            error = output.get("detail")
            st.text(error)
