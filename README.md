# DSLR
## DataScience X Logistic Regression

Projet réalisé en équipe avec Freddy Pupier (https://github.com/pups-enterprise).


The aim of this project is to introduce you to the basic concept behind linear classification based on the Harry Potter's Sorting Hat.
For this project, you will have to create a one-vs-all classification using logistic regression, to sort Hogwarts students into houses.


### Installation

```pip install -r requirements.txt```

### Usage

- Reproduces the pandas .describe() method

```py describe.py dataset_train.csv```

- Plots an histogramm to show the most homogeneous course

```py histogram.py dataset_train.csv```

![hist](https://user-images.githubusercontent.com/40288838/70129625-6470e780-167f-11ea-8894-fdae2c0807cf.PNG)


- Plots the two similar features, use ```-c``` to plot the Pearson's correlation coefficient heatmap

```py scatter_plot.py dataset_train.csv``` or ```py scatter_plot.py dataset_train.csv -c```

![scat_1](https://user-images.githubusercontent.com/40288838/70129631-66d34180-167f-11ea-9b21-f2d1d325352d.PNG)
![scat_2](https://user-images.githubusercontent.com/40288838/70129632-676bd800-167f-11ea-8d30-c02d3fe5c955.PNG)


- Plots the pair plot matrix of all features

```py pair_plot.py dataset_train.csv```

![pair](https://user-images.githubusercontent.com/40288838/70129635-69ce3200-167f-11ea-9007-af846294761b.png)


- Trains the model, use ```-e``` to evaluate the model, i.e slices the dataset into 80% / 20% to split the training and testing part and verify the accuracy of the prediction, use 
```-c``` to compare our model with the scikit-learn one using the accuracy score between the two predictions.

```py train.py dataset_train.csv -e``` or ```py train.py dataset_train.csv -c```

![train_1](https://user-images.githubusercontent.com/40288838/70129640-6d61b900-167f-11ea-9f19-9af0b1cc063c.PNG)
![train_2](https://user-images.githubusercontent.com/40288838/70129641-6d61b900-167f-11ea-9739-c742ab3c1185.PNG)
![train](https://user-images.githubusercontent.com/40288838/70129642-6dfa4f80-167f-11ea-8c7b-f7ef4fca99eb.PNG)


- Make the prediction, use ```-c``` to compare prediction with the scikit-learn model (only if ```-c``` has been used in ```train.py```)

```py predict.py dataset_test.csv -c```

![predict](https://user-images.githubusercontent.com/40288838/70129647-6f2b7c80-167f-11ea-8fb1-58109d59b7ba.PNG)

