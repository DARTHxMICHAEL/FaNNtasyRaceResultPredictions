Mathematical Modeling and Optimization Techniques for Motorsport Outcome Prediction Using Artificial Neural Networks and Genetic Algorithms

This software integrates classical artificial neural networks (ANNs) based on backpropagation and genetic algorithms (GAs) for predicting motorsport race outcomes. In this case using free practice and race results from 2022 and 2023 formula one season. We delve into the mathematical foundations of backpropagation, elucidate the implementation of dynamic learning rate adjustment, describe the mutation process in GAs, and detail the computation of race result scores.

The backpropagation algorithm serves as the cornerstone of training ANNs. Mathematically, it involves iteratively adjusting network weights to minimize the error between predicted and actual outputs. Incorporating a dynamic learning rate enhances the convergence of the backpropagation algorithm. Genetic algorithms mimic the process of natural selection to evolve solutions to optimization problems. In the context of neural network optimization, GAs are employed to explore the solution space and identify optimal network architectures.

To see how well our predictions work, we use a system where we give points to the top drivers in each race. We add up the points for the top n drivers, with the first-place driver getting the most points and so on. The data we use for training just lists the drivers in the order they finished each session. Since there are 20 drivers in each session and four sessions per race week, it's easy to figure out how many times we need to train the model. We start by making sure the data is all in the same scale, then we train the model and see how well it does. We keep track of the errors and which models perform the best so we can use them to create new models.
We found that the best mutation rate for making new models was around 20-30%. But when we tried higher rates, we ended up with more mistakes in the data. So, we settled on 20%. Mutation means randomly changing the weights in the model within a certain range fo [-0,5;0,5]

![excel](https://github.com/DARTHxMICHAEL/FaNNtasyRaceResultPredictions/assets/30693125/c85012e1-9610-41e8-afb8-f207eb25d8a4)

![graph](https://github.com/DARTHxMICHAEL/FaNNtasyRaceResultPredictions/assets/30693125/d625a5b8-86de-4c88-8366-263ce2527110)

When we look at the results, we have to round the numbers from the model's output, which can sometimes cause mistakes. Also, because we have a small amount of training data, our results are just okay. After trying nearly 5000 models, the best one got 95 points, but it had 33 mistakes. Then, after one round of changes with a 20% mutation rate, it got 131 points with 20 mistakes. Finally, after another round with the same method, it got 140 points with 21 mistakes.
Even though a perfect score would be 240 points, we're not sure if our model is really understanding the data beyond what it's seen before. Our training data was limited, and our model might be getting stuck in local solutions. To improve it, we might need more data or more details about the races, like the track and weather conditions.

Math behind the scene.:
![IMG_20240224_180229385~2](https://github.com/DARTHxMICHAEL/FaNNtasyRaceResultPredictions/assets/30693125/ce0f3974-67c5-47ab-95d9-5440dc5ac8a9)
![IMG_20240224_180244888~2](https://github.com/DARTHxMICHAEL/FaNNtasyRaceResultPredictions/assets/30693125/27ed472c-9ab7-4731-8196-e753ae9333c2)

