## Local Explanation

Remove or modify some part of the image and see whether this will drastically change the decision of the model.

- Simply replace a block of pixels in the image with pure gray.
- Saliency map: Assume pixels: $\{x_1,\,...,\,x_n\}$, loss: $e$. Add a small $\Delta x$ to a certain pixel $x_i$ and see the change of loss $\Delta e$. If $\Delta e$ is large, then this means $x_i$ is important for decision making.

$$
\frac{\Delta e}{\Delta x}\Rightarrow\frac{\partial e}{\partial x_n}
$$

* Smoother gradient: Add many noises and calculate the average.

- Visualize the latent output vectors of some layers of the neural network.
  - PCA, t-SNE...
- Probing: Train another probe network to test some outputs of the intermediate layers.

## Global Explanation

For example in a convolution neural network, for each filter in each convolutional layer, we can use gradient ascent to find an $x^*$, such that $x^*=argmax\sum_{i,\,j}a_{i,\,j}$, where $a_{i,\,j}$ is a single pixel in the feature map.

### Constraint from Image $x$

If we find an $x^*=argmax\,y$ and add some regularizations on it, we may be able to find an image that is most likely to be classified as $y$ and also recognized by human.

### Constraint from Generator

Add a generator in front of the classifier, instead of finding $x^*=argmax\,y_i$, now we try to find a latent vector $z^*$ such that $z^*=argmax\,y_i$. Then the image is generated through $x^*=G(z^*)$
