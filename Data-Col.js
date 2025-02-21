//Step 1 CHange the number of files to be download in the for loop
//Step 2 Download the file as
let img;
let posenet;
let pose;

let PWClassifier; // Plank and Wallsit classifier -> neural network
let targetLabel; //change when changing the folder

let finished = false;
let TOTAL_IMAGE = 50; //Change accordingly depending on the number of data

// let folder = ["Plank", "Wallsit", "Stand"];
let folder = ["Wallsit"];
// let folder = ["pushdown", "pushup"];

async function keyPressed() {
  if (key == "s") {
    console.log("Saving the data");
    PWClassifier.saveData();
  }
}

function delay(time) {
  return new Promise((resolve, reject) => {
    setTimeout(resolve, time);
  });
}

//Preload Neural network and Posenet
function preload() {
  //nn options
  let options = {
    inputs: 34, //17 pairs, single pose
    outputs: 3, //since the 2 labels- wallsit and plankl
    task: "classification",
    debug: true,
  };
  PWClassifier = ml5.neuralNetwork(options);

  // LOAD TRAINING DATA
  PWClassifier.loadData("Dataset/json2/standwallsit100.json");
}

// function dataReady() {
//   console.log.out("Loaded wallsit dataset");
// }

async function setup() {
  //poseNet options
  for (var fi = 0; fi < folder.length; fi++) {
    for (var i = 100; i < 122; i++) {
      //iterate through all the images in the folder
      await new Promise((next) => {
        //LOADING THE IMAGE
        targetLabel = folder[fi].toLowerCase();
        newImg = folder[fi].toLowerCase() + "(" + i + ").jpg"; //CHANGE WHEN CHANGING THE FOLDER
        path = "Dataset/" + folder[fi] + "/" + newImg; //CHANGE WHEN CHANGING THE FOLDER
        // newImg = "plank(" + i + ").jpg"; //CHANGE WHEN CHANGING THE FOLDER
        // path = "Dataset/Plank/" + newImg; //CHANGE WHEN CHANGING THE FOLDER

        img = loadImage(path);
        console.log("loaded " + path);
        //posenet options
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

        posenet = ml5.poseNet(loaded, posenetOpts);
        //GET KEYPOINT FOR THE LOADED IMAGE
        posenet.on("pose", gotPoses);

        async function finish() {
          await delay(750);
          if (finished) {
            finished = false;
            // console.log('Done with ' + newImg);
            next();
          } else {
            finish(); //wait
          }
        }

        finish();
      });
    }
  }

  console.log("Finished all the images, saving the data...");
  PWClassifier.saveData();
}

function loaded() {
  posenet.singlePose(img);
}

function gotPoses(results) {
  if (results.length > 0) {
    //Dealing with >= 1 valid pose
    for (let index = 0; index < results.length; index++) {
      if (results[index].pose.score > 0.2) {
        pose = results[index].pose;
      }
    }

    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      // console.log(pose.keypoints[i].part);
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      // console.log(i + " x: " + x + " / y: " + y);
      inputs.push(x);
      inputs.push(y);
    }

    let target = [targetLabel];
    console.log(inputs, target);
    PWClassifier.addData(inputs, target);
    console.log("Added");
    finished = true;
  } else {
    console.log("Didn't find pose, moving next");
    finished = true;
  }
}
