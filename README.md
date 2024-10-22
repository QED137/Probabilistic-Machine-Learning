# Probabilistic-Machine-Learning
This repository contains solutions to the exercises from the Probabilistic Machine Learning course at Tübingen University. The exercises cover a range of topics in probabilistic machine learning, which uses probability theory to model uncertainty in data and predictions.
 ![Bayesian Inference](BaysianInference.png)

## Two Methods of Learning:
| Statistical Learning Theory | Probabilistic Learning |
|---|---|
|formulate a loss-function ad hoc, mapping data to predictions/decisions. Then carefully analyse to show that, under some external as asumptions, this model has certain desirable properties| Carefully formulate and critique a generative model. Inference is then uniquely determined by Bayes’ theorem. No need to question or analyse the paradigm over and over again. 
|mathematical analysis in the foreground (often in the asymptotic or large-number limit) |numerical & computational design in the foregrou (the right model may be intractable) |
|statements about errors tend to focus on the worst-case|structured and extensive quantification of uncertainty by the posterior, often core motivation |

# A Comparison between two learning methods:

## Probabilistic Machine Learning

Probabilistic machine learning refers to a subset of machine learning methods that use probability theory to model uncertainty in predictions and to handle variability in data. Probabilistic models explicitly represent uncertainty using probabilities, which can be useful for making more robust and interpretable predictions.

### Key Characteristics:

- **Bayesian Framework**: Many probabilistic machine learning techniques are based on Bayesian statistics, which provide a coherent framework for updating beliefs in the presence of new data.

- **Uncertainty Quantification**: Probabilistic models provide a natural way to quantify uncertainty in predictions, which is critical in many applications such as risk assessment and decision making under uncertainty.

- **Generative Models**: Probabilistic models often describe how data is generated in terms of a stochastic process. This generative approach allows for better understanding and simulation of data.

- **Examples**:
  - Bayesian networks
  - Hidden Markov Models (HMM)
  - Gaussian Mixture Models (GMM)
  - Latent Dirichlet Allocation (LDA)
  - Probabilistic graphical models
  - Bayesian linear regression
## Statistical Learning
 (This is just for definition sake.)
Statistical learning involves a set of tools for understanding data. These tools can be used for tasks such as prediction (forecasting future data points), inference (drawing conclusions from data), and pattern recognition. The primary focus is on developing models that capture the underlying structure of the data and making predictions based on that.

### Key Characteristics:

- **Model-Based Approach**: Statistical learning often relies on models to represent the relationships between variables. These models can be parametric (assuming a specific form for the model) or non-parametric (making fewer assumptions about the form).

- **Emphasis on Inference**: In addition to prediction, statistical learning places a strong emphasis on inference, understanding how changes in predictors affect the response variable, and quantifying uncertainty in estimates.

- **Data-Driven**: The models are trained using data, and their performance is evaluated based on how well they fit the training data and generalize to new data.

- **Examples**:
  - Linear regression
  - Logistic regression
  - Generalized linear models
  - Support Vector Machines (SVM)
  - Principal Component Analysis (PCA)
  - k-nearest neighbors (k-NN)


## Comparison and Integration

- **Statistical Learning vs. Probabilistic Machine Learning**:
  - **Focus**: Statistical learning often focuses more on prediction and inference, while probabilistic machine learning emphasizes uncertainty quantification and generative modeling.
  - **Approach**: Statistical learning methods may or may not use probabilistic frameworks, whereas probabilistic machine learning explicitly uses probabilities to represent uncertainty and model data.

- **Integration**: Many modern machine learning techniques integrate concepts from both statistical learning and probabilistic methods. For example, a model might use statistical learning techniques to make predictions while using probabilistic methods to quantify the uncertainty of those predictions.

## Practical Application:

- **Model Selection**: Choose statistical learning methods when the primary goal is prediction accuracy and interpretability, and probabilistic methods when uncertainty quantification and handling variability in data are crucial.
- **Hybrid Models**: Combine techniques from both fields to leverage the strengths of each. For instance, use a probabilistic framework to model uncertainty and a statistical learning algorithm for accurate predictions.
# Course information.
This repository includes solutions to the exercises given in the Probabilistic Machine Learning course at Department of Computer Science, Cluster of Excellence Machine Learning,Tübingen AI Center, Tübingen University. The course covers theoretical and practical aspects of probabilistic approaches in machine learning, focusing on robust and interpretable predictions.

[1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

[2] Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
