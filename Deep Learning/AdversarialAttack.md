# Human Attack

Denote the correct answer as $y$; the original input and output of the network is $x_0,\,\hat{y_0}$; the input and output after being attacked is $x_a,\,\hat{y_a}$.

$diff(x_0,\,x_a)$ can be measured using the L2-norm or the L-$\infty$ norm.

## Non-Targeted

Objective: Find an $x_a$ with a difference that is imperceptible to the naked eye from the original input. The difference between the ground truth $y$ and the output $\hat{y_a}$ with this $x_a$ is maximized.

$$
x^*=\mathop{\arg\min}\limits_{{diff(x_0,\,x_a)<\epsilon}}L(x) \\
L(x)=-e(y,\,\hat{y_a})
$$

## Targeted

Objective: Find an $x_a$ with a difference that is imperceptible to the naked eye from the original input. Maximize the difference between the ground truth $y$ and the output $\hat{y_a}$ with this $x_a$, and maximize the similarity between the target $y_{target}$ and the $\hat{y_a}$.

$$
x^*=\mathop{\arg\min}\limits_{{diff(x_0,\,x_a)<\epsilon}}L(x) \\
L(x)=-e(y,\,\hat{y_a})+e(y_{target},\,\hat{y_a})
$$

## Approach

### White Box Attack

We have the neural network and we can do gradient descent using the parameter of the network we want to attack.

- Gradient Descent: If $diff(x_0,$ $x^t)$$>\epsilon$, fix $x^t$ with the maximum $x_0+\epsilon$ on the direction of $g$.

=======================

Start from original image $x_0$

For t=1 to T:

    $x^t=x^{t-1}-\lambda g$

    if$diff(x_0, x^t) > \epsilon$:

    $x^t=max(x_0+\epsilon)$

end

=======================

- Fast Gradient Sign Method: Make sure one single update to the maximum update threshold.

$$
g=\begin{bmatrix}
sign(\frac{\partial L}{\partial x_1}|x=x^{t-1}) \\
sign(\frac{\partial L}{\partial x_2}|x=x^{t-1}) \\
... \\
sign(\frac{\partial L}{\partial x_n}|x=x^{t-1})
\end{bmatrix}=
\begin{bmatrix}
1 \\
-1 \\
... \\
1
\end{bmatrix} \\
\lambda=\epsilon
$$

### Black Box Attack

We don't know or have the access to the parameters of the model we want to attack, but we know what data were used to train this model. We can use the data to train a *proxy network*, and we attack this proxy network.

## Training Time Attack

Backdoor in the Model: [arxiv.org/pdf/1804.00792.pdf](https://arxiv.org/pdf/1804.00792.pdf)

# Defense

## Passive Defense

Add a filter, which has barely no effect on original image, but effectively reduce the harm of attack signal.

- Randomization: Apply some augmentation methods to the image before entering the model.

## Proactive Defense

Another way of data augmentation

Given the training set $X=\{(x_1,\,y_1),\,...,\,(x_n,\,y_n)\}$.

Train the model using $X$.

For n in 1:N:

    Generate adversarial sample$\tilde{x_n}$ using an attack algorithm.

Train the model using $\tilde{X}=\{(\tilde{x_1},\,y_1),\,...,\,(\tilde{x_N},\,y_N)\}$.
