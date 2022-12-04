import { tf, Task, data, TaskProvider } from "..";

export const soulbot: TaskProvider = {
  getTask(): Task {
    return {
      taskID: "soulbot",
      displayInformation: {
        taskTitle: "Soulbot",
        summary: {
          preview:
            "Test our platform by using a publicly available <b>tabular</b> dataset. <br><br> Download the passenger list from the Titanic shipwreck here: <a class='underline text-primary-dark dark:text-primary-light' href='https://github.com/epfml/disco/raw/develop/example_training_data/titanic_train.csv'>titanic_train.csv</a> (more info <a class='underline text-primary-dark dark:text-primary-light' href='https://www.kaggle.com/c/titanic'>here</a>). <br> This model predicts the type of person most likely to survive/die in the historic ship accident, based on their characteristics (sex, age, class etc.).",
          overview:
            "We all know the unfortunate story of the Titanic: this flamboyant new transatlantic boat that sunk in 1912 in the North Atlantic Ocean. Today, we revist this tragedy by trying to predict the survival odds of the passenger given some basic features.",
        },
        model:
          "The current model does not normalize the given data and applies only a very simple pre-processing of the data.",
        tradeoffs:
          "We are using a small model for this task: 4 fully connected layers with few neurons. This allows fast training but can yield to reduced accuracy.",
        dataFormatInformation:
          "This model takes as input a CSV file with 12 columns. The features are general information about the passenger (sex, age, name, etc.) and specific related Titanic data such as the ticket class bought by the passenger, its cabin number, etc.<br><br>pclass: A proxy for socio-economic status (SES)<br>1st = Upper<br>2nd = Middle<br>3rd = Lower<br><br>age: Age is fractional if less than 1. If the age is estimated, it is in the form of xx.5<br><br>sibsp: The dataset defines family relations in this way:<br>Sibling = brother, sister, stepbrother, stepsister<br>Spouse = husband, wife (mistresses and fiancés were ignored)<br><br>parch: The dataset defines family relations in this way:<br>Parent = mother, father<br>Child = daughter, son, stepdaughter, stepson<br>Some children travelled only with a nanny, therefore parch=0 for them.<br><br>The first line of the CSV contains the header:<br> PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked<br><br>Each susequent row contains the corresponding data.",
        dataExampleText:
          "Below one can find an example of a datapoint taken as input by our model. In this datapoint, the person is young man named Owen Harris that unfortunnalty perished with the Titanic. He boarded the boat in South Hamptons and was a 3rd class passenger. On the testing & validation page, the data should not contain the label column (Survived).",
        dataExample: [
          { columnName: "Input", columnData: "Hello" },
          { columnName: "Output", columnData: "Bonjour" },
        ],
        headers: ["Input", "Output"],
      },
      trainingInformation: {
        modelID: "soulbot-model",
        epochs: 20,
        roundDuration: 10,
        validationSplit: 0,
        batchSize: 30,
        // TODO: tokenize
        // preprocessingFunctions: [data.TextPreprocessing.Tokenize],
        preprocessingFunctions: [],
        modelCompileData: {
          optimizer: "rmsprop",
          loss: "meanSquaredError",
          metrics: ["meanSquaredError"],
        },
        dataType: "tabular",
        inputColumns: ["Input"],
        outputColumns: ["Output"],
        scheme: "Federated", // secure aggregation not yet implemented for FeAI
        noiseScale: undefined,
        clippingRadius: undefined,
      },
    };
  },

  async getModel(): Promise<tf.LayersModel> {
    // assume the input and outputs are already tokenized
    const numEncoderTokens = 32;
    const latentDim = 128;
    const numDecoderTokens = 32;

    const encoderInputs = tf.layers.input({
      shape: [null, numEncoderTokens] as number[],
      name: "encoderInputs",
    });

    const encoder = tf.layers.lstm({
      units: latentDim,
      returnState: true,
      name: "encoderLstm",
    });
    const [, stateH, stateC] = encoder.apply(
      encoderInputs
    ) as tf.SymbolicTensor[];
    // We discard `encoder_outputs` and only keep the states.
    const encoderStates = [stateH, stateC];

    // Set up the decoder, using `encoder_states` as initial state.
    const decoderInputs = tf.layers.input({
      shape: [null, numDecoderTokens] as number[],
      name: "decoderInputs",
    });
    // We set up our decoder to return full output sequences,
    // and to return internal states as well. We don't use the
    // return states in the training model, but we will use them in inference.
    const decoderLstm = tf.layers.lstm({
      units: latentDim,
      returnSequences: true,
      returnState: true,
      name: "decoderLstm",
    });

    const [decoderOutputs] = decoderLstm.apply([
      decoderInputs,
      ...encoderStates,
    ]) as tf.Tensor[];

    const decoderDense = tf.layers.dense({
      units: numDecoderTokens,
      activation: "softmax",
      name: "decoderDense",
    });

    const decoderDenseOutputs = decoderDense.apply(
      decoderOutputs
    ) as tf.SymbolicTensor;

    const model = tf.model({
      inputs: [encoderInputs, decoderInputs],
      outputs: decoderDenseOutputs,
      name: "seq2seqModel",
    });

    return model;
  },
};
