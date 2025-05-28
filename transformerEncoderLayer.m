function enc_output = transformerEncoderLayer(encoderOutput, pad_idx, d_k, d_v, numHeads, params, encoderInputArray)
enc_output = MultiHeadAttention(encoderOutput, pad_idx, d_k, d_v, numHeads, params, encoderInputArray);
enc_output = permute(enc_output, [3, 1, 2]);
enc_output = enc_output + encoderOutput;
enc_output = LayerNorm(enc_output, params);
ffc_output = PositionwiseFeedForward(enc_output, params);
ffc_output = ffc_output + enc_output;
enc_output = LayerNorm(ffc_output, params);
end