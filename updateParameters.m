function params = updateParameters(params, gradients)
    params.t = params.t + 1;
    lr_t = params.lr;
    params.m.embeddingMatrix = params.beta1 * params.m.embeddingMatrix + (1 - params.beta1) * gradients.embeddingMatrix;
    params.v.embeddingMatrix = params.beta2 * params.v.embeddingMatrix + (1 - params.beta2) * (gradients.embeddingMatrix.^ 2);
    m_hat = params.m.embeddingMatrix / (1 - params.beta1^params.t);
    v_hat = params.v.embeddingMatrix / (1 - params.beta2^params.t);
    params.embeddingMatrix = params.embeddingMatrix - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    params.m.modelDecoder1.embeddingMatrix = params.beta1 * params.m.modelDecoder1.embeddingMatrix + (1 - params.beta1) * gradients.modelDecoder1.embeddingMatrix;
    params.v.modelDecoder1.embeddingMatrix = params.beta2 * params.v.modelDecoder1.embeddingMatrix + (1 - params.beta2) * (gradients.modelDecoder1.embeddingMatrix.^ 2);
    m_hat = params.m.modelDecoder1.embeddingMatrix / (1 - params.beta1^params.t);
    v_hat = params.v.modelDecoder1.embeddingMatrix / (1 - params.beta2^params.t);
    params.modelDecoder1.embeddingMatrix = params.modelDecoder1.embeddingMatrix - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    attention_fields = {'Wq', 'Wk', 'Wv', 'Wo'};
    feedforward_fields = {'W1', 'b1', 'W2', 'b2'};
    norm_fields = {'gamma', 'beta'};
    fc_fields = {'W', 'b'};
    fcEncoder_fields = {'Weights', 'Bias'};
    fcDecoder_fields = {'Weights', 'Bias'};
    lstmDecoder = {'InputWeights', 'RecurrentWeights', 'Bias'};
    for i = 1:numel(fcEncoder_fields)
        field = fcEncoder_fields{i};
        params.m.fcEncoder.(field) = params.beta1 * params.m.fcEncoder.(field) + (1 - params.beta1) * gradients.fcEncoder.(field);
        params.v.fcEncoder.(field) = params.beta2 * params.v.fcEncoder.(field) + (1 - params.beta2) * (gradients.fcEncoder.(field)).^2;
        m_hat = params.m.fcEncoder.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.fcEncoder.(field) / (1 - params.beta2^params.t);
        params.fcEncoder.(field) = params.fcEncoder.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    for i = 1:numel(fcDecoder_fields)
        field = fcDecoder_fields{i};
        params.m.fcDecoder.(field) = params.beta1 * params.m.fcDecoder.(field) + (1 - params.beta1) * gradients.fcDecoder.(field);
        params.v.fcDecoder.(field) = params.beta2 * params.v.fcDecoder.(field) + (1 - params.beta2) * (gradients.fcDecoder.(field)).^2;
        m_hat = params.m.fcDecoder.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.fcDecoder.(field) / (1 - params.beta2^params.t);
        params.fcDecoder.(field) = params.fcDecoder.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    for i = 1:numel(lstmDecoder)
        field = lstmDecoder{i};
        params.m.lstmDecoder.(field) = params.beta1 * params.m.lstmDecoder.(field) + (1 - params.beta1) * gradients.lstmDecoder.(field);
        params.v.lstmDecoder.(field) = params.beta2 * params.v.lstmDecoder.(field) + (1 - params.beta2) * (gradients.lstmDecoder.(field)).^2;
        m_hat = params.m.lstmDecoder.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.lstmDecoder.(field) / (1 - params.beta2^params.t);
        params.lstmDecoder.(field) = params.lstmDecoder.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    for i = 1:numel(attention_fields)
        field = attention_fields{i};
        if any(isnan(gradients.encoder.(field)(:))) || any(isinf(gradients.encoder.(field)(:)))
            error(['Gradient.encoder.', field, ' contains NaN or Inf']);
        end
        params.m.encoder.(field) = params.beta1 * params.m.encoder.(field) + (1 - params.beta1) * gradients.encoder.(field);
        params.v.encoder.(field) = params.beta2 * params.v.encoder.(field) + (1 - params.beta2) * (gradients.encoder.(field)).^2;
        m_hat = params.m.encoder.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.encoder.(field) / (1 - params.beta2^params.t);
        if any(isnan(m_hat(:))) || any(isinf(m_hat(:)))
            error(['m_hat.encoder.', field, ' contains NaN or Inf']);
        end
        if any(isnan(v_hat(:))) || any(isinf(v_hat(:)))
            error(['v_hat.encoder.', field, ' contains NaN or Inf']);
        end
        params.encoder.(field) = params.encoder.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
        if any(isnan(params.encoder.(field)(:))) || any(isinf(params.encoder.(field)(:)))
            error(['params.encoder.', field, ' contains NaN or Inf after update']);
        end
    end
    for i = 1:numel(feedforward_fields)
        field = feedforward_fields{i};
        if any(isnan(gradients.encoder.(field)(:))) || any(isinf(gradients.encoder.(field)(:)))
            error(['Gradient.encoder.', field, ' contains NaN or Inf']);
        end
        params.m.encoder.(field) = params.beta1 * params.m.encoder.(field) + (1 - params.beta1) * gradients.encoder.(field);
        params.v.encoder.(field) = params.beta2 * params.v.encoder.(field) + (1 - params.beta2) * (gradients.encoder.(field)).^2;
        m_hat = params.m.encoder.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.encoder.(field) / (1 - params.beta2^params.t);
        if any(isnan(m_hat(:))) || any(isinf(m_hat(:)))
            error(['m_hat.encoder.', field, ' contains NaN or Inf']);
        end
        if any(isnan(v_hat(:))) || any(isinf(v_hat(:)))
            error(['v_hat.encoder.', field, ' contains NaN or Inf']);
        end
        params.encoder.(field) = params.encoder.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
        if any(isnan(params.encoder.(field)(:))) || any(isinf(params.encoder.(field)(:)))
            error(['params.encoder.', field, ' contains NaN or Inf after update']);
        end
    end
    for i = 1:numel(attention_fields)
        field = attention_fields{i};
        params.m.decoder.(field) = params.beta1 * params.m.decoder.(field) + (1 - params.beta1) * gradients.decoder.(field);
        params.v.decoder.(field) = params.beta2 * params.v.decoder.(field) + (1 - params.beta2) * (gradients.decoder.(field)).^2;
        m_hat = params.m.decoder.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.decoder.(field) / (1 - params.beta2^params.t);
        params.decoder.(field) = params.decoder.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    for i = 1:numel(attention_fields)
        field = attention_fields{i};
        params.m.enc_dec.(field) = params.beta1 * params.m.enc_dec.(field) + (1 - params.beta1) * gradients.enc_dec.(field);
        params.v.enc_dec.(field) = params.beta2 * params.v.enc_dec.(field) + (1 - params.beta2) * (gradients.enc_dec.(field)).^2;
        m_hat = params.m.enc_dec.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.enc_dec.(field) / (1 - params.beta2^params.t);
        params.enc_dec.(field) = params.enc_dec.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    for i = 1:numel(feedforward_fields)
        field = feedforward_fields{i};
        params.m.enc_dec.(field) = params.beta1 * params.m.enc_dec.(field) + (1 - params.beta1) * gradients.enc_dec.(field);
        params.v.enc_dec.(field) = params.beta2 * params.v.enc_dec.(field) + (1 - params.beta2) * (gradients.enc_dec.(field)).^2;
        m_hat = params.m.enc_dec.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.enc_dec.(field) / (1 - params.beta2^params.t);
        params.enc_dec.(field) = params.enc_dec.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    for i = 1:numel(norm_fields)
        field = norm_fields{i};
        params.m.(field) = params.beta1 * params.m.(field) + (1 - params.beta1) * gradients.(field);
        params.v.(field) = params.beta2 * params.v.(field) + (1 - params.beta2) * (gradients.(field)).^2;
        m_hat = params.m.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.(field) / (1 - params.beta2^params.t);
        params.(field) = params.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    for i = 1:numel(fc_fields)
        field = fc_fields{i};
        if any(isnan(gradients.output.(field)(:))) || any(isinf(gradients.output.(field)(:)))
            error(['Gradient.output.', field, ' contains NaN or Inf']);
        end
        params.m.fullyConnectedLayer.(field) = params.beta1 * params.m.fullyConnectedLayer.(field) + (1 - params.beta1) * gradients.fullyConnectedLayer.(field);
        params.v.fullyConnectedLayer.(field) = params.beta2 * params.v.fullyConnectedLayer.(field) + (1 - params.beta2) * (gradients.fullyConnectedLayer.(field)).^2;
        m_hat = params.m.fullyConnectedLayer.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.fullyConnectedLayer.(field) / (1 - params.beta2^params.t);
        params.fullyConnectedLayer.(field) = params.fullyConnectedLayer.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    for i = 1:numel(fc_fields)
        field = fc_fields{i};
        if any(isnan(gradients.output.(field)(:))) || any(isinf(gradients.output.(field)(:)))
            error(['Gradient.output.', field, ' contains NaN or Inf']);
        end
        params.m.fullyConnectedLayer1.(field) = params.beta1 * params.m.fullyConnectedLayer1.(field) + (1 - params.beta1) * gradients.fullyConnectedLayer1.(field);
        params.v.fullyConnectedLayer1.(field) = params.beta2 * params.v.fullyConnectedLayer1.(field) + (1 - params.beta2) * (gradients.fullyConnectedLayer1.(field)).^2;
        m_hat = params.m.fullyConnectedLayer1.(field) / (1 - params.beta1^params.t);
        v_hat = params.v.fullyConnectedLayer1.(field) / (1 - params.beta2^params.t);
        params.fullyConnectedLayer1.(field) = params.fullyConnectedLayer1.(field) - lr_t * m_hat ./ (sqrt(v_hat) + params.epsilon);
    end
    params.m.output.W = params.beta1 * params.m.output.W + (1 - params.beta1) * gradients.output.W;
    params.v.output.W = params.beta2 * params.v.output.W + (1 - params.beta2) * (gradients.output.W .^ 2);
    params.m.output.b = params.beta1 * params.m.output.b + (1 - params.beta1) * gradients.output.b;
    params.v.output.b = params.beta2 * params.v.output.b + (1 - params.beta2) * (gradients.output.b .^ 2);
    m_hat_W = params.m.output.W / (1 - params.beta1^params.t);
    v_hat_W = params.v.output.W / (1 - params.beta2^params.t);
    m_hat_b = params.m.output.b / (1 - params.beta1^params.t);
    v_hat_b = params.v.output.b / (1 - params.beta2^params.t);
    params.output.W = params.output.W - lr_t * m_hat_W ./ (sqrt(v_hat_W) + params.epsilon);
    params.output.b = params.output.b - lr_t * m_hat_b ./ (sqrt(v_hat_b) + params.epsilon);
end