function dlZ = modelEncoder1(encoderEmbeddingsArray, decoderEmbeddingsArray, params, numBlocks, d_k, d_v, numHeads, pad_idx, encoderInputArray, decoderInputArray)
encoderOutput = encoderEmbeddingsArray;
similarity_threshold = 0.90;

for i = 1:numBlocks
    encoderOutput = transformerEncoderLayer(encoderOutput, pad_idx, d_k, d_v, numHeads, params, encoderInputArray);
    if exist('encoderOutput1', 'var')
        sim = cosine_similarity(encoderOutput, encoderOutput1);
        if sim > similarity_threshold
            break;
        end
    end
    encoderOutput1 = encoderOutput;
end

decoderOutput = decoderEmbeddingsArray;
for i = 1:numBlocks
    dec_output = DecoderAttention(decoderOutput, d_k, numHeads, params, decoderInputArray, pad_idx);
    dec_output = permute(dec_output, [3, 1, 2]);
    dec_output = dec_output + decoderOutput;
    dec_output = LayerNorm(dec_output, params);
    decoderOutput = EncoderDecoderAttention(encoderOutput, dec_output, numHeads, pad_idx, d_k, d_v, params, encoderInputArray);
    decoderOutput = permute(decoderOutput, [3, 1, 2]);
    decoderOutput = decoderOutput + dec_output;
    decoderOutput = LayerNorm(decoderOutput, params);
    ffc_output = PositionwiseFeedForward2(decoderOutput, params);
    ffc_output = ffc_output + decoderOutput;
    decoderOutput = LayerNorm(ffc_output, params);
    if exist('decoderOutput1', 'var')
        sim = cosine_similarity(decoderOutput, decoderOutput1);
        if sim > similarity_threshold
            break;
        end
    end
    decoderOutput1 = decoderOutput;
end

[batchSize, ~, embeddingDim] = size(decoderOutput);
sequenceLengths = sum(decoderInputArray ~= pad_idx, 2);
globalRepresentation = zeros(batchSize, embeddingDim, 'like', decoderOutput);
for i = 1:batchSize
    t = sequenceLengths(i);
    if t > 0
        globalRepresentation(i, :) = decoderOutput(i, t, :);
    else
        globalRepresentation(i, :) = zeros(1, embeddingDim, 'like', decoderOutput);
    end
end

globalRepresentation = globalRepresentation';
weights = params.fcEncoder.Weights;
bias = params.fcEncoder.Bias;
dlZ = fullyconnect(globalRepresentation, weights, bias, 'DataFormat', 'CB');
end















