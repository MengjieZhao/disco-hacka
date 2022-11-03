import { tf, data } from '../..'

export class NodeTabularLoader extends data.TabularLoader<string> {
  loadTabularDatasetFrom (source: string, csvConfig: Record<string, unknown>): tf.data.CSVDataset {
    const prefix = 'file://'
    if (source.slice(0, 7) !== prefix) {
      source = prefix + source
    }
    return tf.data.csv(source, csvConfig)
  }
}
