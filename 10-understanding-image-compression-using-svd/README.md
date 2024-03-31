Singular Value Decomposition (SVD) for Image Compression
This repository provides a sample code demonstrating the use of Singular Value Decomposition (SVD) for image compression.

Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three parts:
	SVD(X) = U Σ VT
	U and V are unitary (orthogonal) matrices, meaning UUT = UTU = I and the same for V.
	Σ is a diagonal matrix with non-negative values arranged in hierarchical order.

How SVD is Used for Image Compression
By using SVD, less relevant features can be eliminated by ignoring smaller Σ values. This helps in compressing the image while preserving its essential features.

Code
The provided code computes the SVD of an image matrix and plots the image again by including the most relevant 5, 20, and 100 features.

Installation:
To run this tool, ensure you have numpy and matplotlib installed. You can install it using pip:
pip install matplotlib