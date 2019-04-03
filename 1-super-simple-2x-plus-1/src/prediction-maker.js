const tf = require("@tensorflow/tfjs");
require('@tensorflow/tfjs-node');

const TRAINED_MODEL_LOCATION = 'file://./src/pretrained-models/trained-model-20190403T21:05:48.848Z.json/model.json';

(async () => {

    const trainedModelLocation = TRAINED_MODEL_LOCATION
    // Load one of our pre-trained models
    const model = await tf.loadLayersModel(trainedModelLocation);

    console.log('predictioning y value for x of 20:')
    
    // Run inference with predict().
    model.predict(tf.tensor([20], [1, 1])).print()
    
})()