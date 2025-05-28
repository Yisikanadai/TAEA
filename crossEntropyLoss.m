function loss = crossEntropyLoss(output, decoderTarget, pad_idx)
output = permute(output, [2,3,1]);
[batch_size, seq_len, vocab_size] = size(output);
output = reshape(output, [batch_size*seq_len, vocab_size]);
decoderTarget = decoderTarget(:);
mask = (decoderTarget ~= pad_idx);
if ~any(mask)
    error('No valid tokens for loss calculation');
end
validTarget = decoderTarget(mask);
validOutput = output(mask, :);
epsilon = 1e-8;
logProb = log(validOutput + epsilon);
validTarget = min(max(validTarget, 1), vocab_size);
rows = (1:length(validTarget))';
indices = sub2ind(size(logProb), rows, double(validTarget));
correctLogProb = logProb(indices);
totalLoss = -sum(correctLogProb);
numValidTokens = sum(mask);
loss = totalLoss / numValidTokens;
end    