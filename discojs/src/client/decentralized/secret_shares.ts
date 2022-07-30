import { List } from 'immutable'

import * as tf from '@tensorflow/tfjs'
import * as crypto from 'crypto'

import { Weights } from '../..'

const maxSeed: number = 2 ** 47
/*
Return Weights object that is difference of two weights object
 */
export function subtractWeights (w1: Weights, w2: Weights): Weights {
  if (w1.length !== w2.length) {
    throw new Error('Weights not of the same number of tensors')
  }

  const sub: Weights = []
  for (let i = 0; i < w1.length; i++) {
    sub.push(tf.sub(w1[i], w2[i]))
  } return sub
}

/*
Return sum of multiple weight objects in an array, returns weight object of sum
 */
export function sum (summands: List<Weights>): Weights {
  const summedWeights: Weights = new Array<tf.Tensor>()
  let tensors: Weights = new Array<tf.Tensor>() // list of different sized tensors of 0
  const shapeOfWeights: Weights = summands.get(0) ?? []
  for (let j = 0; j < shapeOfWeights.length; j++) { // add each tensor separately over the number of summands
    for (let i = 0; i < summands.size; i++) {
      const modelUpdate: Weights = summands.get(i) ?? []
      tensors.push(modelUpdate[j])
    }
    summedWeights.push(tf.addN(tensors))
    tensors = new Array<tf.Tensor>()
  }
  return summedWeights
}

/*
Return Weights in the remaining share once N-1 shares have been constructed (where N is number of ready clients)
 */
export function lastShare (currentShares: Weights[], secret: Weights): Weights {
  if (currentShares.length === 0) {
    throw new Error('Need at least one current share to be able to subtract secret from')
  }
  const currentShares2 = List<Weights>(currentShares)
  const last: Weights = subtractWeights(secret, sum(currentShares2))
  return last
}

/*
Generate N additive shares that aggregate to the secret weights array (where N is number of ready clients)
 */
export function generateAllShares (secret: Weights, nParticipants: number, maxShareValue: number): List<Weights> {
  const shares: Weights[] = []
  for (let i = 0; i < nParticipants - 1; i++) {
    shares.push(generateRandomShare(secret, maxShareValue))
  }
  shares.push(lastShare(shares, secret))
  const sharesFinal = List<Weights>(shares)
  return sharesFinal
}

/*
generates one share in the same shape as the secret that is populated with values randomly chosend from
a uniform distribution between (-maxShareValue, maxShareValue).
 */
export function generateRandomShare (secret: Weights, maxShareValue: number): Weights {
  const share: Weights = []
  const seed: number = crypto.randomInt(maxSeed)
  for (const t of secret) {
    share.push(
      tf.randomUniform(
        t.shape, -maxShareValue, maxShareValue, 'float32', seed)
    )
  }
  return share
}
