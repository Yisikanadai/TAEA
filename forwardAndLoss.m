function [loss, gradients] = forwardAndLoss(encoderEmbeddings, decoderEmbeddings, params, numBlocks, d_k, d_v, numHeads, dropout, pad_idx, encoderInput, decoderInput, decoderTarget, aco_train)
output = transformer_forward(encoderEmbeddings, decoderEmbeddings, params, numBlocks, d_k, d_v, numHeads, dropout, pad_idx, encoderInput, decoderInput, aco_train);
loss = crossEntropyLoss(output, decoderTarget, pad_idx);
if any(isnan(loss(:))) || any(isinf(loss(:)))
    error('Loss calculation contains NaN or Inf');
end
loss = dlarray(loss);
gradients = dlgradient(loss, params);
if any(isnan(gradients.encoder.Wk(:))) || any(isinf(gradients.encoder.Wk(:)))
    error('gradients.encoder.Wk contains NaN or Inf');
end
end    