# Math 522 Project
Damian Anderson, Whitney Anderson, Erika Ibarra, Bryce Lunceford, Paul Smith, and Sebastian Valencia

## Project Description

Suppose that we generate data according to some process:
$$
    y_i = f(x_i) + \varepsilon_i
$$
where $f$ is a deterministic function and $\varepsilon_i$ is a random error term. We will do the following:

1. Train a neural network with parameters $\theta$ to predict $y_i$ from $x_i$. Call it $\hat{f}_\theta(x_i)$.
2. Let $X$ be a random variable that is distributed like the data points $x_i$.
3. Compare the distribution of $f(X) - \hat{f}_\theta(X)$ to the distribution of $\varepsilon_i$.
4. If the two distributions are similar, this could explain why double descent is observed.
5. We will try this with a bunch of different neural network architectures, functions $f$ and distributions over $\varepsilon_i$.