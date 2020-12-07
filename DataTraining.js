let PWClassifier;

function setup() {
  let options = {
    inputs: 34,
    //CHANGE THE OUTPUT ACCORDINGLY
    outputs: 2,
    task: "classification",
    debug: true,
    learningRate: 0.1,
  };
  PWClassifier = ml5.neuralNetwork(options);
  PWClassifier.loadData("Dataset/JSON/squats.json", dataReady);
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
    batchSize: 20,
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
