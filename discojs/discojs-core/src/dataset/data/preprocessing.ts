import { tf, Task } from "../..";

type PreprocessImage = (image: tf.TensorContainer) => tf.TensorContainer;
type TokenizedInput = (text: tf.TensorContainer) => tf.TensorContainer;

export type Preprocessing = ImagePreprocessing;

export interface ImageTensorContainer extends tf.TensorContainerObject {
  xs: tf.Tensor3D | tf.Tensor4D;
  ys: tf.Tensor1D | number | undefined;
}

// export interface TextTensorContainer extends tf.TensorContainerObject {
//   xs: tf.Tensor2D;
//   ys: tf.Tensor2D;
// }

export enum ImagePreprocessing {
  Normalize = "normalize",
  Resize = "resize",
}

export enum TextPreprocessing {
  Tokenize = "tokenized",
}

export function getPreprocessImage(task: Task): PreprocessImage {
  const preprocessImage: PreprocessImage = (
    tensorContainer: tf.TensorContainer
  ): tf.TensorContainer => {
    // TODO unsafe cast, tfjs does not provide the right interface
    const info = task.trainingInformation;
    let { xs, ys } = tensorContainer as ImageTensorContainer;
    if (info.preprocessingFunctions?.includes(ImagePreprocessing.Normalize)) {
      xs = xs.div(tf.scalar(255));
    }
    if (
      info.preprocessingFunctions?.includes(ImagePreprocessing.Resize) &&
      info.IMAGE_H !== undefined &&
      info.IMAGE_W !== undefined
    ) {
      xs = tf.image.resizeBilinear(xs, [info.IMAGE_H, info.IMAGE_W]);
    }
    return {
      xs,
      ys,
    };
  };
  return preprocessImage;
}

export function getTokenizedInput(task: Task): TokenizedInput {
  const tokenizedInput: TokenizedInput = (
    tensorContainer: tf.TensorContainer
  ): tf.TensorContainer => {
    // TODO unsafe cast, tfjs does not provide the right interface
    const info = task.trainingInformation;
    // let { xs, ys } = tensorContainer as TextTensorContainer;
    // if (info.preprocessingFunctions?.includes(TextPreprocessing.Tokenize)) {
    //   xs = xs.div(tf.scalar(255));

    //   // Vectorize the data.
    //   const inputCharacterList = [...xs].sort();
    //   const targetCharacterList = [...ys].sort();

    //   const numEncoderTokens = inputCharacterList.length;
    //   const numDecoderTokens = targetCharacterList.length;

    //   // Math.max() does not work with very large arrays because of the stack limitation
    //   const maxEncoderSeqLength = 32;
    //   const maxDecoderSeqLength = 32;

    //   console.log("Number of unique input tokens:", numEncoderTokens);
    //   console.log("Number of unique output tokens:", numDecoderTokens);
    //   console.log("Max sequence length for inputs:", maxEncoderSeqLength);
    //   console.log("Max sequence length for outputs:", maxDecoderSeqLength);

    //   const inputTokenIndex = inputCharacterList.reduce(
    //     (prev, curr, idx) => ((prev[curr] = idx), prev),
    //     {} as { [char: string]: number }
    //   );
    //   const targetTokenIndex = targetCharacterList.reduce(
    //     (prev, curr, idx) => ((prev[curr] = idx), prev),
    //     {} as { [char: string]: number }
    //   );

    //   Save the token indices to file.
    //   const metadataJsonPath = path.join(args.artifacts_dir, "metadata.json");

    //   if (!fs.existsSync(path.dirname(metadataJsonPath))) {
    //     mkdirp.sync(path.dirname(metadataJsonPath));
    //   }

    //   const metadata = {
    //     input_token_index: inputTokenIndex,
    //     target_token_index: targetTokenIndex,
    //     max_encoder_seq_length: maxEncoderSeqLength,
    //     max_decoder_seq_length: maxDecoderSeqLength,
    //   };

    //   const encoderInputDataBuf = tf.buffer<tf.Rank.R3>([
    //     xs.length,
    //     maxEncoderSeqLength,
    //     numEncoderTokens,
    //   ]);
    //   const decoderInputDataBuf = tf.buffer<tf.Rank.R3>([
    //     xs.length,
    //     maxDecoderSeqLength,
    //     numDecoderTokens,
    //   ]);
    //   const decoderTargetDataBuf = tf.buffer<tf.Rank.R3>([
    //     xs.length,
    //     maxDecoderSeqLength,
    //     numDecoderTokens,
    //   ]);

    //   for (const [i, [inputText, targetText]] of zip(
    //     xs,
    //     ys
    //   ).entries() as IterableIterator<[number, [string, string]]>) {
    //     for (const [t, char] of inputText.split("").entries()) {
    //       // encoder_input_data[i, t, input_token_index[char]] = 1.
    //       encoderInputDataBuf.set(1, i, t, inputTokenIndex[char]);
    //     }

    //     for (const [t, char] of targetText.split("").entries()) {
    //       // decoder_target_data is ahead of decoder_input_data by one timestep
    //       decoderInputDataBuf.set(1, i, t, targetTokenIndex[char]);
    //       if (t > 0) {
    //         // decoder_target_data will be ahead by one timestep
    //         // and will not include the start character.
    //         decoderTargetDataBuf.set(1, i, t - 1, targetTokenIndex[char]);
    //       }
    //     }
    //   }
    //   const x_enc = encoderInputDataBuf.toTensor();
    //   const y_dec = decoderTargetDataBuf.toTensor();
    //   return {
    //     x_enc,
    //     y_dec,
    //   };
    // }
  };
  return tokenizedInput;
}
