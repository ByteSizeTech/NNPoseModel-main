// ml5.js: Pose Classification
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/learning/ml5/7.2-pose-classification.html
// https://youtu.be/FYgYyq-xqAw

// All code: https://editor.p5js.org/codingtrain/sketches/JoZl-QRPK

// Separated into three sketches
// 1: Data Collection: https://editor.p5js.org/codingtrain/sketches/kTM0Gm-1q
// 2: Model Training: https://editor.p5js.org/codingtrain/sketches/-Ywq20rM9
// 3: Model Deployment: https://editor.p5js.org/codingtrain/sketches/c5sDNr8eM

let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "none";

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();

  let posenetOpts = {
    architecture: "ResNet50",
    // imageScaleFactor: 0.3,
    outputStride: 32,
    // flipHorizontal: false,
    minConfidence: 0.5,
    maxPoseDetections: 1,
    minPartConfidence: 0.5,
    scoreThreshold: 0.5,
    nmsRadius: 20,
    detectionType: "single",
    inputResolution: 256,
    multiplier: 0.75,
    quantBytes: 2,
  };

  poseNet = ml5.poseNet(video, posenetOpts, modelLoaded);
  poseNet.on("pose", gotPoses);
  let options = {
    inputs: 34, //17 pairs, single pose
    outputs: 2, //since the 2 labels- plankWallsitModel and plankl
    task: "classification",
    debug: true,
  };
  brain = ml5.neuralNetwork(options);
  // PUSHUP MODEL
  const modelInfo = {
    model: "plankWallsitModel/model.json",
    metadata: "plankWallsitModel/model_meta.json",
    weights: "plankWallsitModel/model.weights.bin",
  };
  // const modelInfo = {
  //   model: "squatModel/model.json",
  //   metadata: "squatModel/model_meta.json",
  //   weights: "squatModel/model.weights.bin",
  // };

  brain.load(modelInfo, brainLoaded);
}

function brainLoaded() {
  console.log("pose classification ready!");
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) {
  console.log(results);
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label.toUpperCase();
  }
  console.log(results[0].confidence);
  classifyPose();
}

function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}

function modelLoaded() {
  console.log("poseNet ready");
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }
  }
  pop();

  fill(255, 0, 255);
  noStroke();
  textSize(150);
  textAlign(CENTER, CENTER);
  text(poseLabel, width / 2, height / 2);
}
