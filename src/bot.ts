import './fetch-polyfill'
import {VertexAI, ChatSession} from '@google-cloud/vertexai'
import {PredictionServiceClient, helpers} from '@google-cloud/aiplatform'
import {info, warning} from '@actions/core'
// import pRetry from 'p-retry'
import {VertexAIOptions, Options} from './options'
import {TokenLimits} from './limits'

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
    if (vertexaiOptions.model.startsWith('gemini')) {
      this.api = new VertexAIAPI(options, vertexaiOptions)
    } else {
      this.api = new AIPlatformAPI(options, vertexaiOptions)
    }
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

interface ChatAPI {
  // eslint-disable-next-line no-unused-vars
  sendMessage(message: string): Promise<string>
}

class VertexAIAPI implements ChatAPI {
  private readonly options: Options
  private readonly model: string
  private readonly chatSession: ChatSession

  constructor(options: Options, vertexaiOptions: VertexAIOptions) {
    this.options = options
    this.model = vertexaiOptions.model
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
    if (this.options.debug) {
      info(`request to model (${this.model}):\n${message}`)
    }
    const result = await this.chatSession.sendMessage(message)
    if (this.options.debug) {
      const dump = JSON.stringify(result, null, 2)
      info(`response from model (${this.model}): ${dump}`)
    }
    return result.response.candidates[0].content.parts[0].text || ''
  }
}

class AIPlatformAPI implements ChatAPI {
  private readonly options: Options
  private readonly model: string
  private readonly tokenLimits: TokenLimits
  private readonly client: PredictionServiceClient

  private readonly endpoint: string
  private readonly context: string
  private messageHistory: {author: string; content: string}[] = []

  constructor(options: Options, vertexaiOptions: VertexAIOptions) {
    this.options = options
    this.model = vertexaiOptions.model
    this.tokenLimits = vertexaiOptions.tokenLimits
    this.client = new PredictionServiceClient({
      apiEndpoint: `${options.vertexaiLocation}-aiplatform.googleapis.com`
    })

    this.endpoint = `projects/${options.vertexaiProjectID}/locations/${options.vertexaiLocation}/publishers/google/models/${vertexaiOptions.model}`
    this.context = `${options.systemMessage}
IMPORTANT: Entire response must be in the language with ISO code: ${options.language}
`
  }

  async sendMessage(message: string): Promise<string> {
    this.messageHistory.push({author: 'user', content: message})
    const parameters = {
      temperature: this.options.vertexaiModelTemperature,
      maxOutputTokens: this.tokenLimits.responseTokens,
      topP: this.options.vertexaiModelTopP,
      topK: this.options.vertexaiModelTopK
    }
    const prompt = {
      context: this.context,
      messages: this.messageHistory
    }
    const request = {
      endpoint: this.endpoint,
      parameters: helpers.toValue(parameters),
      instances: [helpers.toValue(prompt)!]
    }

    if (this.options.debug) {
      const dump = JSON.stringify(request, null, 2)
      info(`request to model (${this.model}): ${dump}}`)
    }

    const [response] = await this.client.predict(request)

    if (this.options.debug) {
      const dump = JSON.stringify(response, null, 2)
      info(`response from model (${this.model}): ${dump}`)
    }

    const responseText =
      response?.predictions?.[0].structValue?.fields?.candidates?.listValue
        ?.values?.[0].structValue?.fields?.content?.stringValue || ''

    this.messageHistory.push({author: 'model', content: responseText})

    return responseText
  }
}
