---
layout: post
title: Linear Algebra Basics for ML - Vector Spaces and Transformations
date: 2022-02-07
description: Linear Algebra 5 - Mathematics for Machine Learning
tags: ml ai linear-algebra math
math: true
categories: machine-learning math math-for-ml
# thumbnail: assets/img/linalg_banner.png
giscus_comments: true
related_posts: true
chart:
  plotly: true
toc:
  beginning: true
---



## The Need for Vector Spaces and Transformations

When we work with machine learning and data science, we often deal with data in various forms, such as images, text, or numerical tables. The real challenge is to represent and manipulate this data efficiently. This is where the concept of vector spaces and transformations comes into play. In simple terms, vector spaces allow us to represent data, and transformations help us modify or map this data to different representations.

In machine learning, data is typically represented as vectors in a vector space, and various algorithms manipulate these vectors to extract features, perform operations, or make predictions. Linear transformations help us understand how the data changes when we move it between different coordinate systems. Affine transformations go a step further by adding translations, which are particularly useful in fields like computer vision, where we often deal with translations and rotations of images.

In this blog post, we will explore vector spaces and transformations in detail, linking them to feature engineering and model performance improvements. Let's start by understanding vector spaces and linear transformations.

## Vector Spaces: The Foundation of Data Representation

### What is a Vector Space?

A vector space (or linear space) is a collection of vectors that can be added together and multiplied by scalars, subject to certain rules. Mathematically, a vector space $$V$$ over a field $$F$$ is a set of objects (called vectors), along with two operations: vector addition and scalar multiplication. These operations must satisfy the following axioms:

- **Commutativity of addition**:  
  $$\mathbf{v} + \mathbf{w} = \mathbf{w} + \mathbf{v}$$

- **Associativity of addition**:  
  $$(\mathbf{v} + \mathbf{w}) + \mathbf{u} = \mathbf{v} + (\mathbf{w} + \mathbf{u})$$

- **Additive identity**: There exists a zero vector $$\mathbf{0}$$ such that  
  $$\mathbf{v} + \mathbf{0} = \mathbf{v}$$

- **Additive inverse**: For every vector $$\mathbf{v}$$, there exists $$-\mathbf{v}$$ such that  
  $$\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$$

- **Distributive properties of scalar multiplication**:  
  $$a(\mathbf{v} + \mathbf{w}) = a\mathbf{v} + a\mathbf{w}$$  
  $$(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$$

- **Multiplicative identity of scalars**:  
  $$1 \cdot \mathbf{v} = \mathbf{v}$$

- **Compatibility of scalar multiplication**:  
  $$a(b\mathbf{v}) = (ab)\mathbf{v}$$

In machine learning, we often think of a vector space as the space in which the data exists. For example, if you have a dataset with $$n$$ features, each data point is a vector in $$n$$-dimensional space. The entire dataset is a collection of vectors that form a subspace of that $$n$$-dimensional space.

### Example: Vector Space in Feature Engineering

Let's say we are working with a dataset containing two features, such as height and weight of individuals. These features can be represented as vectors in a 2-dimensional vector space, where each point corresponds to a person’s height and weight. The feature vectors in this space can be manipulated, transformed, and analyzed for machine learning tasks like clustering, regression, and classification.

## Motivation Behind Linear Transformations

A linear transformation is a function that maps a vector in one vector space to a vector in another vector space, preserving the operations of vector addition and scalar multiplication. This is extremely important in machine learning, as many algorithms involve transforming data into new spaces to extract more useful features or to make it easier for the model to learn.

For example, in Principal Component Analysis (PCA), we perform a linear transformation that projects high-dimensional data onto a lower-dimensional subspace, allowing us to visualize or analyze the data more effectively. The key point is that linear transformations preserve the structure of the data in a way that simplifies operations like regression or classification.

## Linear Transformations and Change of Basis

### What is a Change of Basis?

In machine learning, we may need to switch between different coordinate systems, depending on how we want to represent our data. This is where the concept of a change of basis comes in. When we apply a linear transformation to a vector, we can represent it in a new basis (a new set of basis vectors) that might be more convenient for our analysis.

For example, imagine you have a 2D dataset with features height and weight. These features are represented in the standard basis of the 2D plane, i.e., the x-axis and y-axis. However, you may want to rotate the data such that the axes align with the directions of maximum variance. This is a change of basis, where we apply a linear transformation to rotate the data.

Mathematically, if $$A$$ is the transformation matrix and $$\mathbf{x}$$ is a vector in the original basis, then the transformed vector $$\mathbf{x'}$$ in the new basis is given by:  
$$
\mathbf{x'} = A \mathbf{x}
$$

### Practical Example: Change of Basis in PCA

Consider PCA, where we perform an eigen-decomposition of the covariance matrix to find the principal components. These principal components form a new basis, and we can transform the data into this new basis by applying the linear transformation defined by the eigenvectors of the covariance matrix. The new dataset will have the same data points but represented in terms of the directions of maximum variance (the principal components).

This transformation often helps in reducing the dimensionality of the data while retaining the most important features. In essence, PCA is a linear transformation that changes the basis from the original feature space to the space of principal components.

## Affine Transformations: Beyond Linear Mappings

### What is an Affine Transformation?

An affine transformation is a linear transformation followed by a translation. This means that affine transformations include not just rotations, scaling, and shearing, but also shifts in space. Mathematically, an affine transformation can be represented as:  
$$
\mathbf{x'} = A \mathbf{x} + \mathbf{b}
$$
where $$A$$ is the linear transformation matrix, $$\mathbf{x}$$ is the original vector, $$\mathbf{b}$$ is the translation vector, and $$\mathbf{x'}$$ is the transformed vector.

Affine transformations are extremely useful in image processing and computer vision, where we often need to rotate, scale, or translate images.

### Example: Image Manipulation with Affine Transformations

In computer vision, affine transformations are commonly used for tasks such as rotating or resizing images. Consider a scenario where we want to rotate an image by a certain angle or scale it up or down. These operations can be represented as affine transformations.

```python
import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')

# Define affine transformation matrix (rotation)
angle = 45
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

# Apply affine transformation (rotation)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# Show the result
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

In this example, the image is rotated by 45 degrees around its center, and the transformation is applied using an affine matrix.

### Affine Transformations in Feature Engineering

In feature engineering, affine transformations can be used to modify data features, for example, by scaling or translating them. This can help improve the performance of machine learning models, especially when the features vary widely in magnitude or range.

For instance, we might apply an affine transformation to normalize or standardize data by subtracting the mean (translation) and scaling the data by its standard deviation (scaling). This is a common preprocessing step to improve the performance of algorithms like linear regression or neural networks.

## Application: Data Transformation in Feature Engineering

### Why Transform Data?

In machine learning, feature engineering refers to the process of applying mathematical functions to features to improve model performance. Data transformations, such as scaling, normalizing, and encoding, are common operations used in feature engineering.

- **Scaling**: Many machine learning algorithms, such as k-nearest neighbors or gradient descent, perform better when features are on a similar scale. Affine transformations such as min-max scaling or standardization (subtracting the mean and dividing by the standard deviation) are commonly used to scale features.

- **Dimensionality Reduction**: Techniques like PCA, as we mentioned earlier, involve linear transformations to reduce the dimensionality of data. By transforming the data into a lower-dimensional space, we retain the most important features while discarding noise or less useful features.

- **Non-linear Transformations**: Sometimes, applying non-linear transformations to data, such as log transformations or polynomial features, can help make the data more suitable for machine learning models.

### Example: Data Scaling with an Affine Transformation

Consider a dataset with two features, age and income, that have vastly different scales. Income may range from thousands to millions, while age ranges from 0 to 100. Applying an affine transformation like standardization can help bring both features onto a similar scale:

- Translate each feature by subtracting the mean.
- Scale each feature by dividing by its standard deviation.

Mathematically, we can apply this affine transformation to each feature:  
$$
\mathbf{x'} = \frac{\mathbf{x} - \mu}{\sigma}
$$
where $$\mu$$ is the mean and $$\sigma$$ is the standard deviation of the feature.

In Python, we can perform this transformation using `sklearn`:

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Example dataset
data = {'Age': [23, 45, 34, 50, 29], 'Income': [50000, 120000, 85000, 150000, 65000]}
df = pd.DataFrame(data)

# Apply standardization (affine transformation)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

print(df_scaled)
```

This will standardize both features, bringing them to the same scale, which can help improve the performance of models like linear regression or k-means clustering.

---

In this post, we explored **vector spaces**, **linear transformations**, and **affine transformations**, all of which are foundational concepts in linear algebra that play a critical role in machine learning and data science. From **representing data in vector spaces** to **transforming that data** in meaningful ways, these concepts help us better understand and manipulate data, improving our ability to build and refine machine learning models.

Whether you're working with **feature engineering**, **dimensionality reduction**, or **image transformations**, understanding vector spaces and transformations allows you to tackle complex problems and improve your models' performance.

Remember, the next time you're faced with a complex dataset or need to manipulate features, think about how **vector spaces** and **transformations** can help simplify the problem and enhance your machine learning pipeline.
