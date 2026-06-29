// server.js - OpenAI to NVIDIA NIM API Proxy (FIXED v3)
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

// 🔥 THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = true; 

// Model mapping
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
  'nemo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'kimi-k2.6': 'moonshotai/kimi-k2.6',
  'gpt-oss': 'openai/gpt-oss-120b',
  'diffgem': 'google/diffusiongemma-26b-a4b-it',
  'minimax-m3': 'minimaxai/minimax-m3',
  'stepfun-3.7': 'stepfun-ai/step-3.7-flash'
};

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy' });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    let nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-8b-instruct';
    
    // Base request
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      top_p: 0.95, // NIM default
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };
    
    // 🔥 CRITICAL FIX: Exact NVIDIA NIM Syntax for DeepSeek V4
    const isDeepSeekV4 = nimModel.includes('deepseek-v4');
    
    if (isDeepSeekV4) {
      // NVIDIA NIM requires ONLY 'thinking' and 'reasoning_effort' inside chat_template_kwargs
      // 'enable_thinking' causes errors on some NIM endpoints
      const thinkingParams = {
        chat_template_kwargs: {
          thinking: true,
          reasoning_effort: "high"
        }
      };

      // Apply to extra_body (Required by NIM)
      nimRequest.extra_body = thinkingParams;
    }

    // 🔥 CRITICAL FIX: Increase Timeout to 120s
    // DeepSeek V4 "thinking" can take 30-60s before first token
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 120000 // 120 seconds timeout
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      response.data.pipe(res);
    } else {
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
    console.error('Proxy error:', error.message);
    
    if (error.code === 'ECONNABORTED') {
      console.error('⚠️ TIMEOUT: DeepSeek took too long. Ensure vercel.json maxDuration is set.');
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

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

module.exports = app;

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Proxy running on port ${PORT}`);
  });
}
