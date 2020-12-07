let PWClassifier;

function setup() {
  //SQUATS
  let options = {
    inputs: 34,
    //CHANGE THE OUTPUT ACCORDINGLY
    outputs: 2,
    task: "classification",
    activationHidden: "relu",
    activationOutput: "sigmoid",
    modelMetrics: ["accuracy"],
    modelLoss: "meanSquaredError",
    modelOptimizer: null,
    // layers: [], // custom layers
    debug: true, // determines whether or not to show the training visualization
    learningRate: 0.12,
    // hiddenUnits: 16,
  };

  //PUSHUPS
  // let options = {
  //   inputs: 34,
  //   outputs: 2,
  //   task: "classification",
  //   // layers: [], // custom layers
  //   debug: true, // determines whether or not to show the training visualization
  //   learningRate: 0.12,
  //   // hiddenUnits: 16,
  // };

  //PWS
  //  let options = {
  //   inputs: 34,
  //   outputs: 3,
  //   task: "classification",
  //   debug: true,
  // };

  PWClassifier = ml5.neuralNetwork(options);
  PWClassifier.loadData("Dataset/json2/standwallsit.json", dataReady);
}

async function keyPressed() {
  if (key == "s") {
    console.log("Saving the model");
    PWClassifier.save();
  }
}
function dataReady() {
  console.log(PWClassifier.data);
  PWClassifier.normalizeData();

  //PWS
  // const trainingOptions = {
  //   epochs: 100,
  //   // batchSize: 16,
  // };
  //SQUAT
  const trainingOptions = {
    epochs: 300,
    batchSize: 20,
  };

  //PUSHUP
  // const trainingOptions = {
  //   epochs: 100,
  //   // batchSize: 16,
  // };

  PWClassifier.train(trainingOptions, doneTraining);
}
// function whileTraining() {
//   console.log(`epoch: ${epoch}, loss:${loss}`);
// }
function doneTraining() {
  console.log("model trained");
  PWClassifier.save();
}
