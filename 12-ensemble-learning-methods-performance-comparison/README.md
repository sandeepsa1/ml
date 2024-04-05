Ensemble Learning Techniques - Performance Comparison
This project compares the performance of different Ensemble Learning Techniques on two different datasets: the IRIS dataset and the Wine Quality dataset. The code trains three classifiers (Random Forest, AdaBoost, and Gradient Boosting) on each dataset and evaluates their accuracy and training time.

Although all classifiers exhibit strong performance on the two datasets, introducing a controlled level of noise enables finer discrimination between their accuracies and training times.

Results
The code outputs the accuracy and training time for each classifier on each dataset.
In these scenarios, Random Forest demonstrates superior accuracy, outperforming Gradient Boosting. AdaBoost has the worst accuracy ratings. In terms of speed, Random Forest also exhibits the best performance, followed by AdaBoost.

Requirements
Python 3.x
scikit-learn
Install the required dependencies using:
pip install scikit-learn

Copy code
git clone https://github.com/sandeepsa1/ml.git
cd 12-ensemble-learning-methods-comparing-performances

License
This project is licensed under the MIT License.