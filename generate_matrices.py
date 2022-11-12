#!/usr/bin/env python3

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


if len(sys.argv) == 2:
    try:
        ratings = int(sys.argv[1])
        if ratings < 2:
            print("Usage: generate_matrices.py [DATA POINTS]\n\tDATA POINTS must be at least 2.", file=sys.stderr)
            sys.exit(3)
    except ValueError:
        print("Usage: generate_matrices.py [DATA POINTS]\n\tDATA POINTS is an integer argument", file=sys.stderr)
        sys.exit(1)
elif len(sys.argv) > 2:
    print("Usage: generate_matrices.py [DATA POINTS]\n\tDATA POINTS is an integer argument", file=sys.stderr)
    sys.exit(2)
else:
    ratings = 20000


amazon_ratings = pd.read_csv("customers_rating.csv")
amazon_ratings_subset = amazon_ratings.head(ratings)
ratings_utility_matrix = amazon_ratings_subset.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)

X = ratings_utility_matrix.T

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

correlation_matrix = np.corrcoef(decomposed_matrix)

X.to_pickle("X.pickle")
correlation_matrix.dump("correlation_matrix.pickle")
