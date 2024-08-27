# Mixture of Expert (MoE)

What is MoE?

- Replace some of the layers (usually FFNs) in the original network with multiple experts (also FFNs).
- A router assigning each tokens to one or more experts to process.

Why do we need MoE?

- It allows us to scale up the model without increasing computation.

How to implement such a router?

- Usually a linear layer to compute the weight of each expert on current token (soft assignment).

Why don't we use hard assignment (directly assigning a token to one expert and set others as zero).

- This may lead to problem during training because batch data may be imbalanced
- The training may be inefficient since the router usually has several preferred experts, these experts will be trained more and the others are not (solved by adding an auxiliary loss, *aux_loss* in *transformers*).

## Gshard

### Ways to Maintain Balanced Load and Efficiency

- Random routing: Pick two experts; Always choose the one with largest probability and the second one is sampled based on the probability.
- Setting expert capacity: All experts will have a upper limit for the number of tokens can be processed, if the two experts both reach their limit, then this token is considered overflowed and directly sent to the next layer through residual connection.
