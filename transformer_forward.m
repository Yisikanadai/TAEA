function output = transformer_forward(encoderEmbeddingsArray, decoderEmbeddingsArray, params, numBlocks, d_k, d_v, numHeads, dropout, pad_idx, encoderInputArray, decoderInputArray, embeddingDim, a)
encoderOutput = encoderEmbeddingsArray;
for i = 1:numBlocks
    encoderOutput = transformerEncoderLayer(encoderOutput, pad_idx, d_k, d_v, numHeads, params, encoderInputArray);
    encoderOutput = dropout1(encoderOutput, dropout);
end
% latentDimension = round(size(encoderOutput, 2) * 0.4);
decoderOutput = decoderEmbeddingsArray;
for i = 1:numBlocks
    dec_output = DecoderAttention(decoderOutput, d_k, numHeads, params, decoderInputArray, pad_idx);
    dec_output = permute(dec_output, [3,1,2]);
    dec_output = dec_output + decoderOutput;
    dec_output = LayerNorm(dec_output, params);
    decoderOutput = EncoderDecoderAttention(encoderOutput, dec_output, numHeads, pad_idx, d_k, d_v, params, encoderInputArray);
    decoderOutput = permute(decoderOutput, [3,1,2]);
    decoderOutput = decoderOutput + dec_output;
    decoderOutput = LayerNorm(decoderOutput, params);
    ffc_output = PositionwiseFeedForward2(decoderOutput, params);
    ffc_output = ffc_output + decoderOutput;
    decoderOutput = LayerNorm(ffc_output, params);
    decoderOutput = dropout1(decoderOutput, dropout);
end
[batchSize,~, embeddingDim] = size(decoderOutput);
sequenceLengths = sum(decoderInputArray ~= pad_idx, 2);
globalRepresentation = zeros(batchSize, embeddingDim, 'like', decoderOutput);
for i = 1:batchSize
    t = sequenceLengths(i);
    if t > 0
        globalRepresentation(i,:) = decoderOutput(i, t, :);
    else
        globalRepresentation(i, :) = zeros(1, embeddingDim, 'like', decoderOutput);
    end
end
globalRepresentation = globalRepresentation';
weights = params.fcEncoder.Weights;
bias = params.fcEncoder.Bias;
dlZ = fullyconnect(globalRepresentation, weights, bias, 'DataFormat', 'CB');
state = struct;
state.HiddenState = dlZ;
state.CellState = zeros(size(dlZ), 'like', dlZ);
output = modelDecoder1(params, decoderInputArray, state);
end