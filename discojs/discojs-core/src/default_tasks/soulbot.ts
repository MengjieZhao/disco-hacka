import { tf, Task, TaskProvider } from "..";

export const soulbot: TaskProvider = {
  getTask(): Task {
    return {
      taskID: "soulbot",
      displayInformation: {
        taskTitle: "Soulbot",
        summary: {
          preview: "Collabrative training for better an AI th",
          overview: "Feed more chat materials to our soulbot",
        },
        model:
          "The current model does not normalize the given data and applies only a very simple pre-processing of the data.",
        tradeoffs:
          "We are using a small model for this task: an encoder-decoder LSTM model. This allows fast training but can yield to reduced performance.",
        dataFormatInformation:
          "This model takes as input a CSV file with 16 columns. The first 8 is the tokenized input and the second 8 is the tokenized output.",
        dataExampleText:
          "Below one can find an example of a datapoint taken as input by our model. Currently it is assumed that the input text is already tokenized",
        dataExample: [
          { columnName: "Input0", columnData: 124 },
          { columnName: "Input1", columnData: 53 },
          { columnName: "Input2", columnData: 5 },
          { columnName: "Input3", columnData: 1999 },
          { columnName: "Input4", columnData: 4 },
          { columnName: "Input5", columnData: 645 },
          { columnName: "Input6", columnData: 1 },
          { columnName: "Input7", columnData: 54 },
          { columnName: "Output0", columnData: 1241 },
          { columnName: "Output1", columnData: 153 },
          { columnName: "Output2", columnData: 52 },
          { columnName: "Output3", columnData: 1599 },
          { columnName: "Output4", columnData: 41 },
          { columnName: "Output5", columnData: 45 },
          { columnName: "Output6", columnData: 11 },
          { columnName: "Output7", columnData: 4 },
        ],
        headers: [
          "Input0",
          "Input1",
          "Input2",
          "Input3",
          "Input4",
          "Input5",
          "Input6",
          "Input7",
          "Output0",
          "Output1",
          "Output2",
          "Output3",
          "Output4",
          "Output5",
          "Output6",
          "Output7",
        ],
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
          metrics: ["accuracy"],
        },
        dataType: "tabular",
        inputColumns: [
          "Input0",
          "Input1",
          "Input2",
          "Input3",
          "Input4",
          "Input5",
          "Input6",
          "Input7",
        ],
        outputColumns: [
          "Output0",
          "Output1",
          "Output2",
          "Output3",
          "Output4",
          "Output5",
          "Output6",
          "Output7",
        ],
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
