import re
import bs4
import random
import requests
import numpy as np
import pandas as pd
from common import cache
from fake_useragent import UserAgent, FakeUserAgentError
from sklearn.decomposition import TruncatedSVD
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request

app = Flask(__name__)
cache.init_app(app=app, config={"CACHE_TYPE": "filesystem",'CACHE_DIR': '/tmp'})

@app.before_first_request
def function_to_run_only_once(): # TODO: How to speed this up? This takes way too long!
    amazon_ratings = pd.read_csv("customers_rating.csv")
    amazon_ratings_subset = amazon_ratings.head(20000)
    ratings_utility_matrix = amazon_ratings_subset.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)

    X = ratings_utility_matrix.T
    X1 = X

    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)

    correlation_matrix = np.corrcoef(decomposed_matrix)

    cache.set("X", X  )  # ughhh
    cache.set("correlation_matrix", correlation_matrix)


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        X = cache.get("X")
        correlation_matrix = cache.get("correlation_matrix")
       
        product_id = request.form["productid"]
        if product_id == "random": # TODO make a button that does this
            product_id = X.index[random.randint(0, len(correlation_matrix))]

        product_ASINs = list(X.index)
        product_index = product_ASINs.index(product_id)

        correlation_product_ID = correlation_matrix[product_index]

        Recommend_list = list(X.index[correlation_product_ID > 0.90])
        Recommend_list.remove(product_id)

        print(str(product_id).center(20, "="))
        print(Recommend_list[:10])

    
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
