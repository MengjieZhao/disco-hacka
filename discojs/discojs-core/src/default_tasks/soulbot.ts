import { tf, Task, data, TaskProvider } from "..";

export const soulbot: TaskProvider = {
  getTask(): Task {
    return {
      taskID: "soulbot",
      displayInformation: {
        taskTitle: "Soulbot",
        summary: {
          preview:
            "Test our platform by using a publicly available <b>tabular</b> dataset. <br><br> Download the passenger list from the Titanic shipwreck here: <a class='underline text-primary-dark dark:text-primary-light' href='https://github.com/epfml/disco/raw/develop/example_training_data/titanic_train.csv'>titanic_train.csv</a> (more info <a class='underline text-primary-dark dark:text-primary-light' href='https://www.kaggle.com/c/titanic'>here</a>). <br> This model learns a sequence to sequence model based on authorized talks between therapists and patients",
          overview: "Feed more chat materials to our soulbot",
        },
        model:
          "The current model does not normalize the given data and applies only a very simple pre-processing of the data.",
        tradeoffs:
          "We are using a small model for this task: 4 fully connected layers with few neurons. This allows fast training but can yield to reduced accuracy.",
        dataFormatInformation:
          "This model takes as input a CSV file with 2 columns. The first is the tokenized input and the second is the tokenized output.",
        dataExampleText:
          "Below one can find an example of a datapoint taken as input by our model. In this datapoint, the person is young man named Owen Harris that unfortunnalty perished with the Titanic. He boarded the boat in South Hamptons and was a 3rd class passenger. On the testing & validation page, the data should not contain the label column (Survived).",
        dataExample: [
          {
            columnName: "Input",
            columnData: "Hello, how are you feel right now?",
          },
          { columnName: "Output", columnData: "I am not feeling well." },
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
          optimizer: "adam",
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
