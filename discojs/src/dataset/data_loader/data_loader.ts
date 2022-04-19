import { Dataset } from '../dataset_builder'
import { Task } from '../../task'

export interface DataConfig { features?: string[], labels?: string[] }
export type Data = {
  dataset: Dataset
  size: number
}

export abstract class DataLoader<Source> {
  protected task: Task

  constructor (task: Task) {
    this.task = task
  }

  abstract load (source: Source, config: DataConfig): Promise<Dataset>

  abstract loadAll (sources: Source[], config: DataConfig): Promise<Data>
}