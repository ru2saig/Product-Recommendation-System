import re
import bs4
import random
import requests
import numpy as np
import pandas as pd
from fake_useragent import UserAgent, FakeUserAgentError
from sklearn.decomposition import TruncatedSVD
from concurrent.futures import ThreadPoolExecutor


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
        return ("404", "404")

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


amazon_ratings = pd.read_csv("customers_rating.csv")
amazon_ratings_subset = amazon_ratings.head(20000)
ratings_utility_matrix = amazon_ratings_subset.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)

X = ratings_utility_matrix.T
X1 = X

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

correlation_matrix = np.corrcoef(decomposed_matrix)


while True:
    print("Product ID: Enter \"random\" for a random product")
    user_inp = input()

    # TODO: user input validation
    if user_inp == "random":
        product_id = X.index[random.randint(0, len(correlation_matrix))]
    else:
        product_id = user_inp

    product_ASINs = list(X.index)
    product_index = product_ASINs.index(product_id)

    correlation_product_ID = correlation_matrix[product_index]

    Recommend_list = list(X.index[correlation_product_ID > 0.90])
    Recommend_list.remove(product_id)
    Recommend_list.insert(0, product_id)

    print(str(product_id).center(20, '='))

    with ThreadPoolExecutor() as executor:
        results = executor.map(download_data, Recommend_list[:10])

        for result in results:
            print(result)
