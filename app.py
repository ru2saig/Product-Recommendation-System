import re
import bs4
import random
import requests
import numpy as np
import pandas as pd
from common import cache
from fake_useragent import UserAgent, FakeUserAgentError
from sklearn.decomposition import TruncatedSVD
from flask import Flask, render_template, request
from concurrent.futures import ThreadPoolExecutor


app = Flask(__name__)
cache.init_app(app=app, config={"CACHE_TYPE": "filesystem",'CACHE_DIR': '/tmp'})

HEADERS = {'authority': 'www.amazon.com',
            'pragma': 'no-cache',
            'cache-control': 'no-cache',
            'dnt': '1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'sec-fetch-site': 'none',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-dest': 'document',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',}

product_image_re = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*")
product_title_re = re.compile("[|-]")


def download_data(ASIN_id):
    """
    Download data from amazon.com by ASIN id.

    Takes a single argument, the ASIN_id Does not use proxies,
    or different ip addresses, or different headers. (yet)
    could get ip banned if run too often
    """
    ASIN = ASIN_id
    URL = f"https://www.amazon.com/exec/obidos/ASIN/{ASIN}"
    try:
        HEADERS['user-agent'] = UserAgent(cache=False).random
    except FakeUserAgentError:
        HEADERS['user-agent'] = 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36'

    webpage = requests.get(URL, headers=HEADERS)

    if not webpage.ok:  # If they're not on amazon.com, they can be on other domains
        print(f"Unable to find info on {ASIN_id} on amazon.com")
        return (404, 404)

    if webpage.status_code > 500:
        print("Request blocked by amazon. Try a proxy")
        return ("Beep", "Boop")

    print(f"Downloading Data of { ASIN_id }...")
    soup = bs4.BeautifulSoup(webpage.content, "lxml")
    title = soup.find("span", attrs={"id":'productTitle'})

    product_name = product_title_re.split(title.string.strip())[0]
    product_image = soup.find("div", attrs={"id" : "imgTagWrapperId"})
    matches = product_image_re.finditer(str(product_image), re.MULTILINE)

    for match in matches:
        url = match.group()
        if "http" in url:
            product_image = url
            break

    print(f"Download for { ASIN_id } complete!")
    return (product_name, product_image)


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
        product_result = download_data(product_id)
        
        # TODO: Building a new dataset should fix this right up
        results_data = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(download_data, Recommend_list[:10])

            for result in results:
                results_data.append(result)
        print(results_data)

        return render_template("index.html", product_result=product_result, results_data=results_data)

    return render_template("index.html", product_result=("",""), results_data=[])


if __name__ == "__main__":
    app.run(debug=True)
