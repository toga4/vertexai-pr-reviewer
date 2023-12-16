import './fetch-polyfill'
import {
  VertexAI,
  ChatSession,
  GenerateContentResult
} from '@google-cloud/vertexai'
import {info, warning} from '@actions/core'
// import pRetry from 'p-retry'
import {VertexAIOptions, Options} from './options'

// define type to save parentMessageId and conversationId
export interface Ids {
  parentMessageId?: string
  conversationId?: string
}

export class Bot {
  private readonly api: ChatSession
  private readonly options: Options

  constructor(options: Options, vertexaiOptions: VertexAIOptions) {
    this.options = options
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

    const systemMessage = `${options.systemMessage}
IMPORTANT: Entire response must be in the language with ISO code: ${options.language}
`
    this.api = generativeModel.startChat({
      history: [
        {role: 'user', parts: [{text: systemMessage}]},
        {role: 'model', parts: [{text: options.replyForSystemMessage}]}
      ]
    })
  }

  chat = async (message: string): Promise<string> => {
    try {
      return await this.chat_(message)
    } catch (e: unknown) {
      if (e instanceof Error) {
        warning(`Failed to chat: ${e}, backtrace: ${e.stack}`)
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

    info(`----- MESSAGE START -----
${message}
----- MESSAGE END -----`)
    let response: GenerateContentResult | undefined

    try {
      // response = await pRetry(
      //   async () => {
      //     return this.api!.sendMessage(message)
      //   },
      //   {retries: this.options.vertexaiRetries}
      // )
      response = await this.api!.sendMessage(message)
    } catch (e: unknown) {
      if (e instanceof Error) {
        warning(
          `failed to send message to vertexai: ${e}, backtrace: ${e.stack}`
        )
      } else {
        warning(`failed to send message to vertexai: ${e}`)
      }
    }
    const end = Date.now()
    info(`response: ${JSON.stringify(response, null, 2)}`)
    info(
      `vertexai sendMessage (including retries) response time: ${
        end - start
      } ms`
    )
    let responseText = ''
    if (response != null) {
      responseText = response.response.candidates[0].content.parts[0].text || ''
    } else {
      warning('vertexai response is null')
    }
    if (this.options.debug) {
      info(`vertexai responses: ${responseText}`)
    }
    return responseText
  }
}
