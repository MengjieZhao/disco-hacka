import express from 'express'

import { tf, client, Task, TaskID } from '@epfml/discojs'

import { Server } from '../server'
import { ParamsDictionary } from 'express-serve-static-core'
import { ParsedQs } from 'qs'

import { Map } from 'immutable'
import msgpack from 'msgpack-lite'
import WebSocket from 'ws'

import messages = client.decentralized.messages
type PeerID = client.decentralized.PeerID

export class Decentralized extends Server {
  // maps peerIDs to their respective websockets so peers can be sent messages by their IDs
  private readyClientsBuffer: Map<TaskID, Set<PeerID>> = Map()
  private clients: Map<PeerID, WebSocket> = Map()
  // increments with addition of every client, server keeps track of clients with this and tells them their ID
  private clientCounter: PeerID = 0

  protected get description (): string {
    return 'DeAI Server'
  }

  protected buildRoute (task: Task): string {
    return `/${task.taskID}`
  }

  protected sendConnectedMsg (ws: WebSocket): void {
    const msg: messages.clientConnectedMessage = {
      type: messages.type.clientConnected
    }
    ws.send(msgpack.encode(msg))
  }

  public isValidUrl (url: string | undefined): boolean {
    const splittedUrl = url?.split('/')

    return (splittedUrl !== undefined && splittedUrl.length === 3 && splittedUrl[0] === '' &&
      this.isValidTask(splittedUrl[1]) && this.isValidWebSocket(splittedUrl[2]))
  }

  protected initTask (task: Task, model: tf.LayersModel): void {}

  protected handle (
    task: Task,
    ws: import('ws'),
    model: tf.LayersModel,
    req: express.Request<
    ParamsDictionary,
    any,
    any,
    ParsedQs,
    Record<string, any>
    >
  ): void {
    const minimumReadyPeers = task.trainingInformation?.minimumReadyPeers ?? 3
    const peerID: PeerID = this.clientCounter++
    this.clients = this.clients.set(peerID, ws)
    // send peerID message
    const msg: messages.PeerID = {
      type: messages.type.PeerID,
      id: peerID
    }
    console.info('peer', peerID, 'joined', task.taskID)

    if (!this.readyClientsBuffer.has(task.taskID)) {
      this.readyClientsBuffer.set(task.taskID, new Set<PeerID>())
    }

    ws.send(msgpack.encode(msg), { binary: true })

    // how the server responds to messages
    ws.on('message', (data: Buffer) => {
      try {
        const msg: unknown = msgpack.decode(data)
        if (
          !messages.isMessageToServer(msg) &&
          !messages.isPeerMessage(msg)
        ) {
          console.warn('invalid message received:', msg)
          return
        }

        switch (msg.type) {
          case messages.type.Weights: {
            const forwardMsg: messages.Weights = {
              type: messages.type.Weights,
              peer: peerID,
              weights: msg.weights
            }
            const encodedMsg: Buffer = msgpack.encode(forwardMsg)

            // sends message it received to destination
            this.clients.get(msg.peer)?.send(encodedMsg)
            break
          }
          case messages.type.Shares: {
            const forwardMsg: messages.Shares = {
              type: messages.type.Shares,
              peer: peerID,
              weights: msg.weights
            }
            const encodedMsg: Buffer = msgpack.encode(forwardMsg)

            // sends message it received to destination
            this.clients.get(msg.peer)?.send(encodedMsg)
            break
          }
          case messages.type.PeerIsReady: {
            const currentClients: Set<PeerID> =
              this.readyClientsBuffer.get(msg.task) ?? new Set<PeerID>()
            const updatedClients: Set<PeerID> = currentClients.add(peerID)
            this.readyClientsBuffer = this.readyClientsBuffer.set(
              msg.task,
              updatedClients
            )
            // if enough clients are connected, server shares who is connected
            const currentPeers: Set<PeerID> =
              this.readyClientsBuffer.get(msg.task) ?? new Set<PeerID>()
            if (currentPeers.size >= minimumReadyPeers) {
              this.readyClientsBuffer = this.readyClientsBuffer.set(
                msg.task,
                new Set<PeerID>()
              )
              const readyPeerIDs: messages.PeersForRound = {
                type: messages.type.PeersForRound,
                peers: Array.from(currentPeers)
              }
              for (const peerID of currentPeers) {
                // send peerIds to everyone in readyClients
                this.clients.get(peerID)?.send(msgpack.encode(readyPeerIDs))
              }
            }
            break
          }
          case messages.type.PartialSums: {
            const forwardMsg: messages.PartialSums = {
              type: messages.type.PartialSums,
              peer: peerID,
              partials: msg.partials
            }
            const encodedMsg: Buffer = msgpack.encode(forwardMsg)

            // sends message it received to destination
            this.clients.get(msg.peer)?.send(encodedMsg)
            break
          }
        }
      } catch (e) {
        console.error('when processing WebSocket message:', e)
      }
    })
  }
}
