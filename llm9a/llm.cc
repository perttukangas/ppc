#include "llm.h"

using namespace std;

void multi_head_attention(
    const LLamaConfig &config,
    const LLamaLayer &layer,
    float *keys,
    float *values,
    float *activation,
    int position);

void matmul(float *out, const float *x, const float *w, int n, int d);

void llm(LLamaConfig config, LLamaParameters params, const vector<token_t> &tokens, vector<float> &logits)
{
    using namespace utils;

    vector<vector<float>> keys(config.n_layers, vector<float>(config.seq_len * config.dim));
    vector<vector<float>> values(config.n_layers, vector<float>(config.seq_len * config.dim));

    for (unsigned position = 0; position < tokens.size(); ++position)
    {
        // a few convenience variables
        int dim = config.dim;
        int hidden_dim = config.hidden_dim;
        int token_id = (int)tokens[position];

        // initialize the activation by the embedding of the current token
        vector<float> activation(params.TokenEmbeddingMatrix.begin() + token_id * dim,
                                 params.TokenEmbeddingMatrix.begin() + (token_id + 1) * dim);

        // scratch buffers to use during the computations
        vector<float> buffer(dim);
        vector<float> buffer2(dim);
        vector<float> hidden_buffer(hidden_dim);
        vector<float> hidden_buffer2(hidden_dim);

        // forward all the layers
        for (int l = 0; l < config.n_layers; l++)
        {
            auto &layer = params.LayerWeights[l];

            // attention rmsnorm
            rmsnorm(buffer.data(),
                    activation.data(),
                    layer.rms_attention.data(),
                    dim);

            multi_head_attention(config, layer, keys[l].data(), values[l].data(), buffer.data(), position);

            // final matmul to get the output of the attention
            matmul(buffer2.data(), buffer.data(), layer.out_weight_matrix.data(), dim, dim);

            // residual connection back into x
            for (int i = 0; i < dim; i++)
            {
                activation[i] += buffer2[i];
            }

            rmsnorm(buffer.data(), activation.data(), layer.rms_feed_forward.data(), dim);

            matmul(hidden_buffer.data(), buffer.data(), layer.feed_forward_w1.data(), dim, hidden_dim);
            matmul(hidden_buffer2.data(), buffer.data(), layer.feed_forward_w3.data(), dim, hidden_dim);
            swiglu(hidden_buffer.data(), hidden_buffer.data(), hidden_buffer2.data(), hidden_dim);
            // final matmul to get the output of the ffn
            matmul(buffer.data(), hidden_buffer.data(), layer.feed_forward_w2.data(), hidden_dim, dim);

            // residual connection back into x
            for (int i = 0; i < dim; i++)
            {
                activation[i] += buffer[i];
            }
        }

        // final rmsnorm
        rmsnorm(activation.data(), activation.data(), params.RmsFinal.data(), dim);

        // classifier into logits
        float *current_logits = logits.data() + position * config.vocab_size;
        matmul(current_logits, activation.data(), params.TokenOutputMatrix.data(), dim, config.vocab_size);
    }
}

void matmul(float *out, const float *x, const float *w, int n, int d)
{
#pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        int j = 0;

        for (; j + 3 < n; j += 4)
        {
            sum0 += x[j] * w[i * n + j];
            sum1 += x[j + 1] * w[i * n + j + 1];
            sum2 += x[j + 2] * w[i * n + j + 2];
            sum3 += x[j + 3] * w[i * n + j + 3];
        }

        for (; j < n; j++)
        {
            sum0 += x[j] * w[i * n + j];
        }
        out[i] = sum0 + sum1 + sum2 + sum3;
    }
}

void multi_head_attention(
    const LLamaConfig &config,
    const LLamaLayer &layer,
    float *keys,
    float *values,
    float *activation,
    int position)
{
    using namespace utils;
    int dim = config.dim;
    int head_size = config.head_size();

    // key and value point to the kv cache
    float *key = keys + position * dim;
    float *value = values + position * dim;

    vector<float> query(dim);

    // qkv matmuls for this position.
    matmul(query.data(),
           activation,
           layer.query_weight_matrix.data(),
           dim, dim);

    // key and value are generated directly at the desired position
    // inside the cache
    matmul(key,
           activation,
           layer.key_weight_matrix.data(),
           dim, dim);
    matmul(value,
           activation,
           layer.value_weight_matrix.data(),
           dim, dim);

    rope(config, query.data(), key, position);

    vector<float> attention(config.seq_len);

    // multi-head attention. iterate over all heads
    for (int h = 0; h < config.n_heads; h++)
    {
        // get the query vector for this head
        float *q = query.data() + h * head_size;

        calculate_attention(config, attention.data(), q, position,
                            keys + h * head_size);
        lookup_with_attention(config, attention.data(),
                              activation + h * head_size,
                              position,
                              values + h * head_size);
    }
}