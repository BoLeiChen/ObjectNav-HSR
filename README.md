# ObjectNav-HSR
This repository contains the codes for our paper, which is titled "Socially Aware Object Goal Navigation with Heterogeneous Scene Representation Learning".

# Abstract
Socially aware Object Goal Navigation (ObjectNav) requires robots to understand complex social awareness among humans and the semantic co-occurrence relations among objects. Existing methods usually achieve scene representation by mapping complex Human-Robot-Object (HRO) mutual interactions to the same feature space. However, this homogeneous scene representation may result in losing features' specificity. We argue that humans, robots, and objects have different interaction paradigms with each other and should be represented separately and elaborately. Therefore, a novel Heterogeneous Scene Representation (HSR) learning method is proposed in our work to learn HRO ternary interaction features. In particular, a novel Heterogeneous Graph Attention Network (HGAN) is proposed to exclusively model different interaction paradigms and semantic relations so that they maintain their essential properties. Further, a Deep Reinforcement Learning (DRL) based socially aware ObjectNav strategy is proposed by learning HSR-based environmental state transition and state value estimation. We demonstrate the feasibility and superiority of our method through sufficient comparative studies and baseline tests in challenging domestic crowded scenarios.

# Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library.
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library.
3. Install crowd_sim and crowd_nav into pip:
```
pip install -e.
```
# Dateset
Coming soon...

# Getting Started
Coming soon...
