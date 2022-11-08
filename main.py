import bs4
import re
import requests
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

amazon_ratings = pd.read_csv("customers_rating.csv")
amazon_ratings_subset = amazon_ratings.head(20000)
ratings_utility_matrix = amazon_ratings_subset.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)

X = ratings_utility_matrix.T
X1 = X

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

correlation_matrix = np.corrcoef(decomposed_matrix)

HEADERS = ({'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'})

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
    
    for item in Recommend_list[:10]:
        ASIN = item
        URL = f"https://www.amazon.com/exec/obidos/ASIN/{ASIN}"
        webpage = requests.get(URL, headers=HEADERS) # TODO: this takes around 3 seconds per request

        if not webpage.ok: # If they're not on amazon.com, they can be on .uk or .ca, etc.
            print(f"Unable to find info on {item} on amazon.com")
            continue

        print(f"Downloading Data of { item }...")

        soup = bs4.BeautifulSoup(webpage.content, "lxml")
        title = soup.find("span", attrs={"id":'productTitle'})

        # compile the regex expressions, though
        re.split("[|-]", title.string.strip()) # do smthing like [|//+] for multiple characters
        product_name = re.split("[|]", title.string.strip())[0]

        product_image = soup.find("div", attrs={"id" : "imgTagWrapperId"})
        regex = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
        matches = re.finditer(regex, str(product_image), re.MULTILINE)

        for match in matches:
            url = match.group()
            if "http" in url:
                product_image = url
                break

        print(product_name, product_image)
