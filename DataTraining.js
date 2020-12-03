let PWClassifier;

function setup() {
  let options = {
    inputs: 34,
    outputs: 2,
    task: "classification",
    debug: true,
  };
  PWClassifier = ml5.neuralNetwork(options);
  PWClassifier.loadData("Dataset/json2/sq104.json", dataReady);
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
  const trainingOptions = {
    epochs: 100,
    // batchSize: 16,
    learningRate: 0.0001,
  };
  PWClassifier.train(trainingOptions, doneTraining);
}
// function whileTraining() {
//   console.log(`epoch: ${epoch}, loss:${loss}`);
// }
function doneTraining() {
  console.log("model trained");
  PWClassifier.save();
}
