let PWClassifier;

function setup() {
  let options = {
    inputs: 34,
    outputs: 3,
    task: "classification",
    debug: true,
  };
  PWClassifier = ml5.neuralNetwork(options);
  PWClassifier.loadData("Dataset/json/pData.json", dataReady);
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
    epochs: 150,
    batchSize: 16,
    learningRate: 0.01,
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
