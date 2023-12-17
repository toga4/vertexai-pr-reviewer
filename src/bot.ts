import './fetch-polyfill'
import {VertexAI, ChatSession} from '@google-cloud/vertexai'
import {info, warning} from '@actions/core'
// import pRetry from 'p-retry'
import {VertexAIOptions, Options} from './options'

// define type to save parentMessageId and conversationId
export interface Ids {
  parentMessageId?: string
  conversationId?: string
}

export class Bot {
  private readonly api: ChatAPI
  private readonly options: Options

  constructor(options: Options, vertexaiOptions: VertexAIOptions) {
    this.options = options
    this.api = new VertexAIAPI(options, vertexaiOptions)
  }

  chat = async (message: string): Promise<string> => {
    try {
      return await this.chat_(message)
    } catch (e: unknown) {
      if (e instanceof Error) {
        warning(`Failed to chat: ${e}, backtrace: ${e.stack}`)
      } else {
        warning(`Failed to chat: ${e}`)
      }
      return ''
    }
  }

  private readonly chat_ = async (message: string): Promise<string> => {
    // record timing
    const start = Date.now()
    if (!message) {
      return ''
    }
    info('----- MESSAGE START -----\n${message}\n----- MESSAGE END -----')
    // response = await pRetry(
    //   async () => {
    //     return this.api!.sendMessage(message)
    //   },
    //   {retries: this.options.vertexaiRetries}
    // )
    const responseText = await this.api!.sendMessage(message)
    const duration = Date.now() - start
    info(`response time (including retries): ${duration} ms`)
    if (!responseText) {
      warning('vertexai response is empty')
    }
    return responseText
  }
}

declare interface ChatAPI {
  sendMessage(message: string): Promise<string>
}

class VertexAIAPI implements ChatAPI {
  private readonly options: Options
  private readonly chatSession: ChatSession

  constructor(options: Options, vertexaiOptions: VertexAIOptions) {
    this.options = options
    const systemMessage = `${options.systemMessage}
IMPORTANT: Entire response must be in the language with ISO code: ${options.language}
`

    const vertexAI = new VertexAI({
      project: options.vertexaiProjectID,
      location: options.vertexaiLocation
    })

    const generativeModel = vertexAI.preview.getGenerativeModel({
      model: vertexaiOptions.model,
      // eslint-disable-next-line camelcase
      generation_config: {
        // eslint-disable-next-line camelcase
        max_output_tokens: vertexaiOptions.tokenLimits.responseTokens,
        temperature: options.vertexaiModelTemperature,
        // eslint-disable-next-line camelcase
        top_p: options.vertexaiModelTopP,
        // eslint-disable-next-line camelcase
        top_k: options.vertexaiModelTopK
      }
    })

    this.chatSession = generativeModel.startChat({
      history: [
        {role: 'user', parts: [{text: systemMessage}]},
        {role: 'model', parts: [{text: options.replyForSystemMessage}]}
      ]
    })
  }

  async sendMessage(message: string): Promise<string> {
    const result = await this.chatSession.sendMessage(message)
    if (this.options.debug) {
      info(`response: ${JSON.stringify(result, null, 2)}`)
    }
    return result.response.candidates[0].content.parts[0].text || ''
  }
}
