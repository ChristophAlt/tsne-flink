# TSNE on Apache Flink

This is an implementation of t-distributed stochastic neighbor embedding (t-SNE) on top of [Apache Flink](https://flink.apache.org).
The implementation is based on the following papers by Laurens van der Maaten and Geoffrey Hinton:

[Visualizing High-Dimensional Data Using t-SNE](http://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)  
[Accelerating t-SNE using Tree-Based Algorithms](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

## Parameters

When running the dataflow with Apache Flink, the following commandline parameters can/must be specified:

### Mandatory

`--input` path to the input matrix in COO format (either dense or sparse)  
`--output` path where the final embedding is written to  
`--dimension` the dimensionality of the input datapoints  
`--knnMethod` which knn method to use (can be "bruteforce", "partition" or "project")  
`--inputDistanceMatrix` signals that the input is a precomputed distance matrix, which is then used for computing the high dimensional affinities  


### Optional

`--executionPlan` stores the dataflow execution plan in "tsne_executionPlan.json"  
`--metric` the metric used for calculating the similarity between datapoints (can be "sqeuclidean", "euclidean" or "cosine")  
`--perplexity` the perplexity used for determining the variance (via binary search) of the high dimensional affinities  
`--nComponents` the dimensionality of the embedding  
`--earlyExaggeration` a constant the affinities are multiplied with intially to separate clusters  
`--learningRate` the initial gradient step size  
`--iterations` the number of gradient iterations  
`--randomState` the seed for the PRNG used to initialize the embedding  
`--neighbors` the number of neighbors used to approximate the high dimensional affinities  
`--initialMomentum` the momentum used during the initial gradient steps (iteration < 20)  
`--finalMomentum` the momentum used during the final gradient steps (iteration > 20  
`--theta` the threshold used for approximating the repulsive forces during gradient computation (lower values give better accuracy but also less performance)  
`--lossFile` path to the file where the loss per iteration is stored  
`--knnIterations` only used when knnMethod == "project", specifies the number of projection steps  
`--knnBlocks` only used when knnMethod == "partition", specifies the number of partitions  
