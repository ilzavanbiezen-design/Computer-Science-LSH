# Computer-Science-LSH

This repository contains a Python implementation of a scalable product duplicate detection method for Web shop data, developed as part of an academic assignment on entity resolution. The goal of the project is to efficiently identify duplicate product offers across different online retailers while avoiding the computational cost of exhaustive pairwise comparison.

The goal of this project is to identify duplicate television product offers across different Web shops. Since comparing all product pairs is computationally infeasible for large datasets, this implementation applies LSH as a blocking technique to efficiently generate candidate pairs. These candidates are then filtered using similarity measures and domain-specific constraints.
Products are represented using a combination of:
- Character 3-grams extracted from product titles
- Structured attributes such as brand, screen size, resolution, and UPC codes

MinHash signatures are constructed to approximate Jaccard similarity, after which LSH is used to significantly reduce the number of comparisons. Duplicate products are identified using a clustering-based approach with union–find.

The entire implementation is contained in a single Python file:
- Data structures and preprocessing:
    - Product data class
    - JSON data loading
    - Text normalization and feature extraction
    - Token construction and IDF computation
- MinHash and LSH:
    - Custom MinHash signature generation
    - Banding and bucketing for LSH
    - Candidate pair generation based on signature similarity
- Duplicate detection and evaluation:
    - Similarity computation using (weighted) Jaccard similarity
    - Hard filtering based on brand, size, and resolution
    - Clustering using union–find
    - Evaluation using pair quality, pair completeness, F1*, precision, recall, and F1-score
    - Bootstrapping for robust evaluation

All functions are defined first. The actual execution of the pipeline only starts when the script is run directly.

To run the code:
- Download the data
- Updata the data path in the main block
- run: python LSH.py
- When executed, the script:
    - Loads and preprocesses the data
    - Builds MinHash signatures
    - Evaluates a baseline method using all pairwise comparisons
    - Runs LSH-based duplicate detection with bootstrapping for multiple LSH thresholds
    - Produces evaluation metrics and plots illustrating the trade-off between efficiency and effectiveness
