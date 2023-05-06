# Regression Model

## Using the Model

RegressionModel.py is the main program to run the regression model.
It has five boolean flags on lines 120-124:

```
trainModel = False
loadModel = True
saveModel = True
testModel = False
testExample = False
```

Setting `trainModel = True` will start the training loop for the model. The model takes a long time to train, so it is best to use the provided weightsr `RegressionModelWeightsFinal.state`.

`loadModel = True` will load the model weights from the file in the variable `LOAD_PATH`.

`saveModel = True` will save the model weights during training to the file `SAVE_PATH`.

`testModel = True` will test the model on sample games included in the folder Games.

`testExample = True` will test the model on the kaggle test set, and produce a csv file that Kaggle can read.

The default behaviour of the file, without training, is to produce plots of the model performance using the holdout parts of the training set.
These plots are stored in the Images/ folder. 
Since the split is random, the holdout set is different each time the model is run.

## How the model was trained

The model was trained for 100 epochs using a base learning rate of $10^{-4}$, and a `ReduceLROnPlateau` scheduler with a reduction of 0.5 to obtain the results shown in the report.
The trained model used a 20% dropout rate as well after all ReLU activation functions to help prevent overfitting.
