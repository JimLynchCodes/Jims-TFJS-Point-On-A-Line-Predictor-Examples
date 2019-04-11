const tf =  require("@tensorflow/tfjs");
 require('@tensorflow/tfjs-node');

const TRAINED_MODEL_LOCATION = 'file://src/pretrained-models/trained-model-20190405T18:34:59.362Z.json/model.json';

(async () => {

    const trainedModelLocation = TRAINED_MODEL_LOCATION
    // Load one of our pre-trained models
    const model = await tf.loadLayersModel(trainedModelLocation);

    console.log('predicting y value for x...')
    
    // Run inference with predict().
    model.predict(tf.tensor([20, 2, 4], [1, 3])).print()
    
})()