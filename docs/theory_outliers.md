# Deep Dive: Activation Outliers in LLMs

*A supplementary guide for [Understanding TurboQuant](article.md)*

When compressing Large Language Models, the most significant mathematical hurdle is not the average size of the numbers, but their distribution. Specifically, we must contend with **Activation Outliers**.

## What are Activation Outliers?

In a standard transformer architecture, the output of attention layers and feed-forward networks (FFNs) produces high-dimensional vectors for every token. If you plot the values of these vectors across all dimensions (channels), you might expect a normal, bell-curve distribution centered around zero. 

Instead, empirical observation reveals a heavy-tailed distribution where a few specific channels consistently exhibit extreme magnitudes—often 10x to 100x larger than the average value in other channels.

### Why do they occur?
Research suggests these outliers are not bugs, but features learned by the model. Certain channels evolve to act as strong indicators for syntactic structure or to pass "no-op" (no operation) signals when the attention mechanism shouldn't update the residual stream significantly. The model relies on these massive values to effectively route information.

## The Quantization Trap

Quantization maps continuous floating-point numbers into a discrete set of bins. The simplest approach, Uniform Quantization, divides the range $[\min(X),\, \max(X)]$ into equally sized intervals.

### The Mathematics of the Failure
Suppose we want to quantize a vector $X$ to $b$ bits. We have $2^b$ available bins.
The step size $\Delta$ is defined as:
$$\Delta = \frac{\max(X) - \min(X)}{2^b - 1}$$

If $X$ contains mostly values between $-1$ and $1$, but has a single outlier at $100$, the range becomes $[-1, 100]$.
For 4-bit quantization ($2^4 = 16$ bins), the step size becomes:
$$\Delta \approx \frac{100 - (-1)}{15} = \frac{101}{15} \approx 6.73$$

The bins are placed at $-1, 5.73, 12.46...$
Every value between $-1$ and $1$ (which constitutes 99% of the important, nuanced data) will be rounded into the exact same bin ($-1$). All the fine-grained information is permanently destroyed just to accommodate the single outlier at $100$.

When this vector is used to compute attention scores ($Q \cdot K$), the loss of precision in the non-outlier channels leads to catastrophic degradation in the model's reasoning capabilities.

![Comparative Clamping Behavior](assets/1_Comparitive_clamping_behaviour_roatated%20quantization_meth.png)

## The Need for a Pre-conditioning Step

Since the model fundamentally requires the information encoded in these outliers, we cannot simply clip them (cap them at a lower value). If we clip the outlier, the model loses its syntax marker. If we keep the outlier, uniform quantization destroys the rest of the vector. 

This paradox necessitates a mathematical transformation that preserves the geometric information of the vector while flattening the distribution—which leads us to the **Principle of Incoherence**.
