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
      stream: stream || true
    };

    console.log(`[${requestId}] Forwarding request to NVIDIA NIM: ${nimModel} (Stream: ${stream})`);

    // Make request to NVIDIA NIM API with a 90-second timeout
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 90000 // 90 seconds timeout (forces error before Vercel 120s CDN timeout)
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
