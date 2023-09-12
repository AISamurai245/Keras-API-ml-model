Train a Machine Learning Model using Prebuilt Dataset with Keras
This README guide will walk you through the process of training a machine learning model using a prebuilt dataset and the Keras API. In this example, we will use the popular MNIST dataset to build a neural network that classifies images.
Prerequisites
Before you begin, make sure you have TensorFlow installed. You can check the TensorFlow version by running the following code:

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

Load a Dataset
We'll start by loading and preparing the MNIST dataset. The pixel values of the images range from 0 through 255. To normalize these values to a range of 0 to 1, divide them by 255.0. This also converts the sample data from integers to floating-point numbers.

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

Build a Machine Learning Model

Next, we will build a neural network model using the tf.keras.Sequential API. This model consists of a Flatten layer, a Dense layer with ReLU activation, a Dropout layer, and another Dense layer with 10 units (our output layer).

model =  tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

For each example, the model returns a vector of logits or log-odds scores, one for each class.

Make Predictions

You can make predictions using the untrained model as follows:

predictions = model(x_train[:1]).numpy()

The tf.nn.softmax function can be used to convert these logits into probabilities for each class.

Define a Loss Function

To train the model, we need to define a loss function. In this example, we use tf.keras.losses.SparseCategoricalCrossentropy:

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

Compile the Model
Before training, configure and compile the model using model.compile. Set the optimizer to 'adam,' the loss function to loss_fn, and specify 'accuracy' as the metric to evaluate the model:

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

Train and Evaluate the Model
Now, you can train the model using the Model.fit method:

model.fit(x_train, y_train, epochs=5)


And evaluate the model's performance using the Model.evaluate method:

model.evaluate(x_test, y_test, verbose=2)

Wrap the Model for Probability Output
If you want your model to return probabilities, you can wrap the trained model and attach the softmax activation to it:

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model(x_test[:5])



Conclusion
Congratulations! You have successfully trained a machine learning model using a prebuilt dataset with the Keras API. This example achieved an accuracy of approximately 98% on the MNIST dataset. You can further fine-tune the model or apply it to your specific image classification tasks.

































