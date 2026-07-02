// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); 
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true; // Set to true to show reasoning with tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping containing all 19 of your custom models
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'gemma': 'google/gemma-4-31b-it',
  'gemma7': 'google/gemma-7b',
  'mistral': 'mistralai/mistral-small-4-119b-2603',
  'opus5.1': 'z-ai/glm-5.1',
  'opus4.7': 'z-ai/glm4.7',
  'deepseek-v4': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4f': 'deepseek-ai/deepseek-v4-flash',
  'nemo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'nemo4b': 'nvidia/nemotron-mini-4b-instruct',
  'kimi-k2.6': 'moonshotai/kimi-k2.6',
  'gpt-oss': 'openai/gpt-oss-120b',
  'diffgem': 'google/diffusiongemma-26b-a4b-it',
  'minimax-m3': 'minimaxai/minimax-m3',
  'stepfun-3.7': 'stepfun-ai/step-3.7-flash',
  'seed': 'bytedance/seed-oss-36b-instruct'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy'
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  const requestId = `req-${Date.now()}`;
  console.log(`[${requestId}] Incoming proxy request for model: ${req.body.model}`);

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Get NIM model mapping
    let nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-8b-instruct';

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    console.log(`[${requestId}] Forwarding request to NVIDIA NIM: ${nimModel} (Stream: ${stream})`);

    // Make request to NVIDIA NIM API with a 90-second timeout
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000 // 300 seconds timeout (forces error before Vercel 120s CDN timeout)
    });

    console.log(`[${requestId}] Received response headers from NVIDIA NIM. Status: ${response.status}`);

    if (stream) {
      // Handle streaming response with Vercel optimizations
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('Content-Encoding', 'none'); // Prevents Vercel buffer compression

      console.log(`[${requestId}] Piping stream chunks to client...`);
      response.data.pipe(res);

      // Log when the stream finishes or gets closed
      response.data.on('end', () => {
        console.log(`[${requestId}] Stream finished successfully.`);
      });

      // Clean up the downstream stream if the user disconnects early
      req.on('close', () => {
        console.log(`[${requestId}] Client disconnected early. Closing stream.`);
        response.data.destroy();
      });
    } else {
      // Transform NIM response to OpenAI format
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices,
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error(`[${requestId}] Proxy error:`, error.message);

    // If it was a timeout, send a clear message back to the client
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return res.status(504).json({
        error: {
          message: 'The downstream model provider (NVIDIA NIM) took too long to respond. The request timed out.',
          type: 'gateway_timeout_error',
          code: 504
        }
      });
    }

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Export for Vercel
module.exports = app;

// Local development
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
  });
}

app.get(['/favicon.ico', '/favicon.png'], (req, res) => {
  res.status(204).end();
});
