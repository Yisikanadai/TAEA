function params = initAdam(embeddingDim, numHeads, d_k, d_v, d_ff, vocab_size, latentDimension)
params = struct();
params.embeddingMatrix = dlarray(xavierInit(embeddingDim, vocab_size));
params.m.embeddingMatrix = dlarray(zeros(size(params.embeddingMatrix)));
params.v.embeddingMatrix = dlarray(zeros(size(params.embeddingMatrix)));
params.modelDecoder1.embeddingMatrix = dlarray(xavierInit(embeddingDim, vocab_size));
params.m.modelDecoder1.embeddingMatrix = dlarray(zeros(size(params.modelDecoder1.embeddingMatrix)));
params.v.modelDecoder1.embeddingMatrix = dlarray(zeros(size(params.modelDecoder1.embeddingMatrix)));

params.encoder.Wq = dlarray(xavierInit(numHeads * d_k, embeddingDim));
params.encoder.Wk = dlarray(xavierInit(numHeads * d_k, embeddingDim));
params.encoder.Wv = dlarray(xavierInit(numHeads * d_v, embeddingDim));
params.encoder.Wo = dlarray(xavierInit(embeddingDim, numHeads * d_v));
params.encoder.W1 = dlarray(initializeHe([d_ff, embeddingDim], embeddingDim));
params.encoder.b1 = dlarray(zeros(d_ff, 1));
params.encoder.W2 = dlarray(initializeHe([embeddingDim, d_ff], d_ff));
params.encoder.b2 = dlarray(zeros(embeddingDim, 1));

params.decoder.Wq = dlarray(xavierInit(numHeads * d_k, embeddingDim));
params.decoder.Wk = dlarray(xavierInit(numHeads * d_k, embeddingDim));
params.decoder.Wv = dlarray(xavierInit(numHeads * d_v, embeddingDim));
params.decoder.Wo = dlarray(xavierInit(embeddingDim, numHeads * d_v));

params.enc_dec.Wq = dlarray(xavierInit(numHeads * d_k, embeddingDim));
params.enc_dec.Wk = dlarray(xavierInit(numHeads * d_k, embeddingDim));
params.enc_dec.Wv = dlarray(xavierInit(numHeads * d_v, embeddingDim));
params.enc_dec.Wo = dlarray(xavierInit(embeddingDim, numHeads * d_v));
params.enc_dec.W1 = dlarray(initializeHe([d_ff, embeddingDim], embeddingDim));
params.enc_dec.b1 = dlarray(zeros(d_ff, 1));
params.enc_dec.W2 = dlarray(initializeHe([embeddingDim, d_ff], d_ff));
params.enc_dec.b2 = dlarray(zeros(embeddingDim, 1));

params.output.W = dlarray(xavierInit(vocab_size, embeddingDim));
params.output.b = dlarray(zeros(1, vocab_size));
params.lr = dlarray(0.001);
outputDim = round(0.4 * embeddingDim);
params.fullyConnectedLayer.W = dlarray(xavierInit(outputDim, embeddingDim));
params.fullyConnectedLayer.b = dlarray(zeros(1, outputDim));
params.fullyConnectedLayer1.W = dlarray(xavierInit(embeddingDim, outputDim));
params.fullyConnectedLayer1.b = dlarray(zeros(1, embeddingDim));

params.beta1 = dlarray(0.9);
params.beta2 = dlarray(0.98);
params.epsilon = dlarray(1e-8);
params.gamma = dlarray(ones(1, 1, embeddingDim));
params.beta = dlarray(zeros(1, 1, embeddingDim));
params.t = dlarray(0);

params.m.encoder = struct('Wq', dlarray(zeros(size(params.encoder.Wq))), ...
    'Wk', dlarray(zeros(size(params.encoder.Wk))), ...
    'Wv', dlarray(zeros(size(params.encoder.Wv))), ...
    'Wo', dlarray(zeros(size(params.encoder.Wo))), ...
    'W1', dlarray(zeros(size(params.encoder.W1))), ...
    'b1', dlarray(zeros(size(params.encoder.b1))), ...
    'W2', dlarray(zeros(size(params.encoder.W2))), ...
    'b2', dlarray(zeros(size(params.encoder.b2))));
params.v.encoder = struct('Wq', dlarray(zeros(size(params.encoder.Wq))), ...
    'Wk', dlarray(zeros(size(params.encoder.Wk))), ...
    'Wv', dlarray(zeros(size(params.encoder.Wv))), ...
    'Wo', dlarray(zeros(size(params.encoder.Wo))), ...
    'W1', dlarray(zeros(size(params.encoder.W1))), ...
    'b1', dlarray(zeros(size(params.encoder.b1))), ...
    'W2', dlarray(zeros(size(params.encoder.W2))), ...
    'b2', dlarray(zeros(size(params.encoder.b2))));
params.m.decoder = struct('Wq', dlarray(zeros(size(params.decoder.Wq))), ...
    'Wk', dlarray(zeros(size(params.decoder.Wk))), ...
    'Wv', dlarray(zeros(size(params.decoder.Wv))), ...
    'Wo', dlarray(zeros(size(params.decoder.Wo))));
params.v.decoder = struct('Wq', dlarray(zeros(size(params.decoder.Wq))), ...
    'Wk', dlarray(zeros(size(params.decoder.Wk))), ...
    'Wv', dlarray(zeros(size(params.decoder.Wv))), ...
    'Wo', dlarray(zeros(size(params.decoder.Wo))));
params.m.enc_dec = struct('Wq', dlarray(zeros(size(params.enc_dec.Wq))), ...
    'Wk', dlarray(zeros(size(params.enc_dec.Wk))), ...
    'Wv', dlarray(zeros(size(params.enc_dec.Wv))), ...
    'Wo', dlarray(zeros(size(params.enc_dec.Wo))), ...
    'W1', dlarray(zeros(size(params.enc_dec.W1))), ...
    'b1', dlarray(zeros(size(params.enc_dec.b1))), ...
    'W2', dlarray(zeros(size(params.enc_dec.W2))), ...
    'b2', dlarray(zeros(size(params.enc_dec.b2))));
params.v.enc_dec = struct('Wq', dlarray(zeros(size(params.enc_dec.Wq))), ...
    'Wk', dlarray(zeros(size(params.enc_dec.Wk))), ...
    'Wv', dlarray(zeros(size(params.enc_dec.Wv))), ...
    'Wo', dlarray(zeros(size(params.enc_dec.Wo))), ...
    'W1', dlarray(zeros(size(params.enc_dec.W1))), ...
    'b1', dlarray(zeros(size(params.enc_dec.b1))), ...
    'W2', dlarray(zeros(size(params.enc_dec.W2))), ...
    'b2', dlarray(zeros(size(params.enc_dec.b2))));
params.m.output = struct('W', dlarray(zeros(size(params.output.W))), ...
    'b', dlarray(zeros(size(params.output.b))));
params.v.output = struct('W', dlarray(zeros(size(params.output.W))), ...
    'b', dlarray(zeros(size(params.output.b))));
params.m.gamma = dlarray(zeros(size(params.gamma)));
params.v.gamma = dlarray(zeros(size(params.gamma)));
params.m.beta = dlarray(zeros(size(params.beta)));
params.v.beta = dlarray(zeros(size(params.beta)));
params.m.fullyConnectedLayer.W = dlarray(zeros(size(params.fullyConnectedLayer.W)));
params.v.fullyConnectedLayer.W = dlarray(zeros(size(params.fullyConnectedLayer.W)));
params.m.fullyConnectedLayer.b = dlarray(zeros(size(params.fullyConnectedLayer.b)));
params.v.fullyConnectedLayer.b = dlarray(zeros(size(params.fullyConnectedLayer.b)));
params.m.fullyConnectedLayer1.W = dlarray(zeros(size(params.fullyConnectedLayer1.W)));
params.v.fullyConnectedLayer1.W = dlarray(zeros(size(params.fullyConnectedLayer1.W)));
params.m.fullyConnectedLayer1.b = dlarray(zeros(size(params.fullyConnectedLayer1.b)));
params.v.fullyConnectedLayer1.b = dlarray(zeros(size(params.fullyConnectedLayer1.b)));

params.fcEncoder.Weights = dlarray(xavierInit(embeddingDim, latentDimension));
params.fcEncoder.Bias = dlarray(xavierInit(latentDimension, 1));
params.lstmDecoder.InputWeights = dlarray(xavierInit(embeddingDim, 4 * latentDimension));
params.lstmDecoder.RecurrentWeights = dlarray(xavierInit(latentDimension, 4 * latentDimension));
params.lstmDecoder.Bias = dlarray(xavierInit(1, 4 * latentDimension));
params.fcDecoder.Weights = dlarray(xavierInit(latentDimension, vocab_size));
params.fcDecoder.Bias = dlarray(xavierInit(vocab_size, 1));

params.m.fcEncoder.Weights = dlarray(zeros(size(params.fcEncoder.Weights)));
params.v.fcEncoder.Weights = dlarray(zeros(size(params.fcEncoder.Weights)));
params.m.fcEncoder.Bias = dlarray(zeros(size(params.fcEncoder.Bias)));
params.v.fcEncoder.Bias = dlarray(zeros(size(params.fcEncoder.Bias)));
params.m.fcDecoder.Weights = dlarray(zeros(size(params.fcDecoder.Weights)));
params.v.fcDecoder.Weights = dlarray(zeros(size(params.fcDecoder.Weights)));
params.m.fcDecoder.Bias = dlarray(zeros(size(params.fcDecoder.Bias)));
params.v.fcDecoder.Bias = dlarray(zeros(size(params.fcDecoder.Bias)));
params.m.lstmDecoder.InputWeights = dlarray(zeros(size(params.lstmDecoder.InputWeights)));
params.v.lstmDecoder.InputWeights = dlarray(zeros(size(params.lstmDecoder.InputWeights)));
params.m.lstmDecoder.RecurrentWeights = dlarray(zeros(size(params.lstmDecoder.RecurrentWeights)));
params.v.lstmDecoder.RecurrentWeights = dlarray(zeros(size(params.lstmDecoder.RecurrentWeights)));
params.m.lstmDecoder.Bias = dlarray(zeros(size(params.lstmDecoder.Bias)));
params.v.lstmDecoder.Bias = dlarray(zeros(size(params.lstmDecoder.Bias)));
end    