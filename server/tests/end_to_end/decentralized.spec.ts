import fs from 'fs/promises'
import path from 'node:path'
import { Server } from 'node:http'
import { Range } from 'immutable'

import {
  node, informant, Task, Disco, tasks, client as clients, WeightsContainer, aggregation
} from '@epfml/discojs-node'

import { getClient, startServer } from '../utils'

describe('end to end decentralized', function () {
  const epsilon: number = 0.001
  this.timeout(50_000)

  let server: Server
  beforeEach(async () => {
    server = await startServer()
  })
  afterEach(() => {
    server?.close()
  })

  it('runs cifar 10 with three clear text decentralized users', async () => {
    await Promise.all([cifar10User(false), cifar10User(false), cifar10User(false)])
  })

  it('runs cifar 10 with three secure decentralized users', async () => {
    await Promise.all([cifar10User(true), cifar10User(true), cifar10User(true)])
  })

  async function cifar10User (secure: boolean): Promise<void> {
    const dir = '../example_training_data/CIFAR10/'
    const files = (await fs.readdir(dir)).map((file) => path.join(dir, file))
    const labels = Range(0, 24).map((label) => (label % 10).toString()).toArray()

    const cifar10: Task = tasks.cifar10.task

    const loaded = await new node.data.NodeImageLoader(cifar10).loadAll(files, { labels: labels })

    const client = secure
      ? await getClient(clients.decentralized.SecAgg, server, cifar10)
      : await getClient(clients.decentralized.ClearText, server, cifar10)

    const disco = new Disco(cifar10, { client })
    await disco.fit(loaded)
  }

  it('decentralized client test one round of clear text weight aggregation', async () => {
    await testWeightSharing(false)
  })

  it('decentralized client test one round of secure weight aggregation', async () => {
    await testWeightSharing(true)
  }
  )

  /*
    Makes client object to connect to server. The input array is the weights that the client will share
    with other ready peers. The input will vary with model architecture and training data. If secure is true,
    the client will implement secure aggregation. If it is false, it will be a clear text client.
     */
  async function makeClient (input: number[], secure: boolean): Promise<WeightsContainer> {
    const cifar10 = tasks.cifar10.task
    const client = secure
      ? await getClient(clients.decentralized.SecAgg, server, cifar10)
      : await getClient(clients.decentralized.ClearText, server, cifar10)
    const weights = WeightsContainer.of(input)
    const trainingInformantCurrent = new informant.DecentralizedInformant(cifar10, 0)
    await client.connect()
    return await client.onRoundEndCommunication(weights, weights, 0, trainingInformantCurrent)
  }

  /*
  Creates three clients with different update values and returns the aggregated update value between all three clients.
  The clients have model dimension of 4 model updates to share, which can be seen as their input parameter in makeClient().
   */
  async function testWeightSharing (secure: boolean): Promise<void> {
    const expected = WeightsContainer.of([0.002, 7, 27, 11])
    const client1 = makeClient([0.001, 3, 40, 10], secure)
    const client2 = makeClient([0.002, 5, 30, 11], secure)
    const client3 = makeClient([0.003, 13, 11, 12], secure)
    const result = await Promise.all([client1, client2, client3])
    aggregation.assertWeightsEqual(result[0], expected, epsilon)
  }

  it('decentralized secure client testing timout', async () => {
    const result = await testTimeOut()
    const expected = WeightsContainer.of([4, 5, 6, 7])
    aggregation.assertWeightsEqual(expected, result, epsilon)
  })

  async function testTimeOut (): Promise<WeightsContainer> {
    return await makeClient([4, 5, 6, 7], true)
  }
})
