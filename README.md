# Imitation Learning for Multi-objectiveOptimization−AlphaMOEA      
In the last decade, a variety of multi-objective
evolutionary algorithms (MOEAs) with specific enhancements
have been developed for solving multi-objective optimization
problems (MOPs). In this paper, unlike MOEAs, we provide
a new artificial intelligence approach to solve MOPs, which
adopts an imitation learning-based end-to-end method, namely
AlphaMOEA. AlphaMOEA is entirely a model composed of
neural networks that mainly follow the architecture of multi-
task learning. It has two training stages, i.e., supervised learning
(SL) and reinforcement learning (RL) stage. In the SL stage,
AlphaMOEA fits the solutions in the decision space provided
by a number of selected MOEAs. Since neural networks in
AlphaMOEA are composed of parameters with high dimensions,
the fitting process can be viewed as a transformation of the
solutions from a low dimensional space into a high dimensional
space. This allows AlphaMOEA to obtain different valuable
knowledge from a perspective of high dimensionality. Then, Al-
phaMOEA is trained in the RL stage to obtain good performance
for MOPs with various problem characteristics in a self-driven
manner. The RL stage relies on several designed components,
including a similarity-based state design to measure the distance
between solutions, an evolution operator-based action set to
provide exploration behavior, and an indicator-guided reward to
produce an incremental evaluation. Experimental results show
that AlphaMOEA can learn the valuable information of the
decision space in the high dimensional representation to achieve
a desirable balance of exploration and exploitation, and further
improve the performance for solving MOPs with various problem
characteristics in a reasonable time.
