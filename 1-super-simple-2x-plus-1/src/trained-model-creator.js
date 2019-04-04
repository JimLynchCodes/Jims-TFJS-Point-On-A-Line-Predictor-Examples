require("@tensorflow/tfjs");
const tf = require('@tensorflow/tfjs-node');

const nowString = (new Date()).toISOString().replace(/-/g,"");
const locationToSaveModel = 'file://./src/pretrained-models/trained-model-' + nowString + '.json'

// Create a sequential model because we watch to build a "linear classification" model
const model = tf.sequential();

// // Add the different "layers" of data points that are thought to influence the outcome.
// "units" represents "the dimentiality of the ouput space".
// "input shape" is how many numbers go into each input. 
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// // Set some config for how to train the model
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// // Generate some synthetic data for training.
// // TODO - use real data from https://www.mysportsfeeds.com
// const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
// const xs = tf.tensor2d([[[1, 1, 1], [1,1,1]], [[2, 2, 2], [2,2,2]], [[3, 3, 3], [3,3,3]], [[4, 4, 4], [4, 4, 4]]], [4, 3]);
// const ys = tf.tensor2d([[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]], [4, 3]);
// const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// vector of input cases
xs = tf.tensor([[0], [1], [2], [5]], [4, 1]);

// vector of output cases
ys = tf.tensor([[1], [3], [5], [11]], [4, 1]);

(async () => {
   await model.fit(xs, ys, {epochs: 1000});
   await model.save(locationToSaveModel)
   console.log('your model has been successfully saved to ' + locationToSaveModel + '!');
})();

