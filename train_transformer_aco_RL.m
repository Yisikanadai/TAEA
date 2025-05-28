clear;
clc;
load train.mat;

numEpochs = 500;
batchSize = 128;
embeddingDim = 100;
pad_idx = 0;
numHeads = 4;
d_k = 25;
d_v = 25;
d_ff = 200;
threshold_loss = 0.5;
vocab_size = data.n + 3;
numBlocks = 4;
dropout = 0.1;
maxGradNorm = 2;
initialLR = 0.0001;
decayRate = 0.98;
numHiddenUnits = 70;
latentDimension = 16;

params = initAdam(embeddingDim, numHeads, d_k, d_v, d_ff, vocab_size, latentDimension);
[encoderInput, decoderInput, decoderTarget, encoderEmbeddings, decoderEmbeddings] = preprocessSeqTrain(aco_train, batchSize, embeddingDim, params);
[encoderEmbeddings, decoderEmbeddings] = addPositionalEncoding2(encoderEmbeddings, decoderEmbeddings);
[encoderEmbeddingsArray, decoderEmbeddingsArray, encoderInputArray, decoderInputArray, decoderTargetArray] = cellToTrainingArrays(encoderEmbeddings, decoderEmbeddings, encoderInput, decoderInput, decoderTarget, pad_idx);

total_samples = size(encoderInputArray, 1);
total_steps = numEpochs * (total_samples / batchSize);
warmup_epochs = floor(0.05 * total_steps);

encoderInputArray = single(encoderInputArray);
decoderInputArray = single(decoderInputArray);
decoderTargetArray = single(decoderTargetArray);
encoderEmbeddingsArray = single(encoderEmbeddingsArray);
decoderEmbeddingsArray = single(decoderEmbeddingsArray);

lossHistory = zeros(numEpochs * floor(size(encoderInputArray, 1) / batchSize), 1);
epochLossHistory = zeros(numEpochs, 1);
iteration = 0;

if canUseGPU
    encoderInputArray = gpuArray(encoderInputArray);
    decoderInputArray = gpuArray(decoderInputArray);
    decoderTargetArray = gpuArray(decoderTargetArray);
    encoderEmbeddingsArray = gpuArray(encoderEmbeddingsArray);
    decoderEmbeddingsArray = gpuArray(decoderEmbeddingsArray);
end

for epoch = 1:numEpochs
    idx = randperm(size(encoderInputArray, 1));
    encoderInputArray = encoderInputArray(idx, :, :);
    decoderInputArray = decoderInputArray(idx, :, :);
    decoderTargetArray = decoderTargetArray(idx, :, :);
    encoderEmbeddingsArray = encoderEmbeddingsArray(idx, :, :);
    decoderEmbeddingsArray = decoderEmbeddingsArray(idx, :, :);
    
    fprintf('Training - Epoch %d/%d\n', epoch, numEpochs);
    epochLoss = 0;
    numIterationsPerEpoch = floor(size(encoderInputArray, 1) / batchSize);
    
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        global_step = (epoch - 1) * numIterationsPerEpoch + i;
        warmup_steps = warmup_epochs * numIterationsPerEpoch;
        params.lr = noam_scheme(iteration, warmup_steps, total_steps);
        
        batchIdx = (i-1)*batchSize + 1 : i*batchSize;
        batchEncoderInput  = encoderInputArray(batchIdx, :, :);
        batchDecoderInput  = decoderInputArray(batchIdx, :, :);
        batchDecoderTarget = decoderTargetArray(batchIdx, :, :);
        encoderEmbedBatch  = encoderEmbeddingsArray(batchIdx, :, :);
        decoderEmbedBatch  = decoderEmbeddingsArray(batchIdx, :, :);
        
        [loss, gradients] = dlfeval(@forwardAndLoss, encoderEmbedBatch, decoderEmbedBatch, params, numBlocks, d_k, d_v, numHeads, dropout, pad_idx, batchEncoderInput, batchDecoderInput, batchDecoderTarget, seq_train);
        params = updateParameters(params, gradients);
        
        lossHistory(iteration) = extractdata(loss);
        epochLoss = epochLoss + extractdata(loss);
    end
    
    avgEpochLoss = epochLoss / numIterationsPerEpoch;
    epochLossHistory(epoch) = avgEpochLoss;
    fprintf('Training - Epoch %d finished - Average Loss: %.4f\n', epoch, avgEpochLoss);
    
    if avgEpochLoss > 0 && avgEpochLoss < threshold_loss
        break;
    end
end

aco_net = params;
save('aco_net.mat', 'aco_net', 'lossHistory');