# Computer-Science-LSH

This repository contains a Python implementation of a scalable product duplicate detection method for Web shop data, developed as part of an academic assignment on entity resolution. The goal of the project is to efficiently identify duplicate product offers across different online retailers while avoiding the computational cost of exhaustive pairwise comparison.

The implementation focuses on television product data and follows a two-stage approach. First, products are represented using a combination of character-level and structured features. Titles are transformed into character 3-grams with high-frequency stop-grams removed, while additional tokens are extracted from commonly available attributes such as brand, screen size, resolution, and UPC codes. Inverse document frequency (IDF) weighting is applied to emphasize informative tokens and reduce the influence of frequent ones.

In the second stage, MinHash signatures are constructed for each product to approximate Jaccard similarity between token sets. These signatures are used as input to a Locality-Sensitive Hashing (LSH) scheme that partitions the signatures into bands and rows, allowing only highly similar products to be considered as candidate pairs. The LSH parameters are automatically derived from a target similarity threshold, enabling systematic control over the trade-off between efficiency and recall.

Candidate pairs produced by LSH are then filtered using domain-specific constraints and a weighted Jaccard similarity measure. Duplicate products are identified through a union–find clustering approach, resulting in groups of matching product offers. The method is evaluated using both pair-level and clustering-based metrics, including pair quality, pair completeness, F1*, precision, recall, and F1-score. Bootstrapping is employed to ensure robust evaluation across multiple train–test splits.

Overall, this project demonstrates that LSH can dramatically reduce the number of comparisons required to detect duplicates, achieving high efficiency while maintaining reasonable effectiveness on real-world Web shop data.
