# Reinforcement Learning

Reinforcement Learning

- Model-based: Learn a world model so that we know the transition probability and the reward, can be solved using dynamic programming.
  - Dynamic Programming:
  - $$
    V(S_{t+1})=E_{\pi_{\theta}}[R_{t+1}+\gamma V_{t+1}]
    $$
- Model-free
  - Value-based: TD learning, SARSA, Q-learning, DQN. Estimate $Q_{\theta}(s, a)$ and greedily pick the one that maximize Q. Deterministic policy.
  - Policy-based:  Optimize $\pi_{\theta}(s, a)$. Stochastic policy.

## Value-based

### TD learning(0)

Estimate value function:

$$
V(s_{t})=V(s_t)+\alpha (R(s_t, a_t)+\gamma V(s_{t+1})-V(s_t))
$$

If TD(n):

$$
V(s_{t})=V(s_t)+\alpha (G_{t:t+n}-V(s_t)) \\
G_{t:t+n}=R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1})+\gamma ^2R(s_{t+2}, a_{t+2})+...+\gamma ^nV(s_{t+n})
$$

### SARSA(0)

Estimate Q-function:

$$
Q(s_t, a_t)=Q(s_t, a_t)+\alpha(R(s_t, a_t)+\gamma Q(s_{t+1}, a_{t+1})-Q(s_t, a_t))
$$


#### On Policy vs. Off Policy:

On-policy: 和环境交互的agent同时也是被优化的agent, SARSA

Off-policy: 是不同的agent, Q-Learning, PPO.

### Q learning

$$
Q(s_t, a_t)=Q(s_t, a_t)+\alpha(R(s_t, a_t)+\gamma (argmax_{a'}Q(s_{t+1}, a'))-Q(s_t, a_t)))
$$
