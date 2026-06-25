function embeddedSeq = embedSequences(seq, embeddingMatrix, embeddingDim)
[batchSize, seqLength] = size(seq);
embeddedSeq = zeros(batchSize, seqLength, embeddingDim);
for i = 1:batchSize
    for j = 1:seqLength
        embeddedSeq(i, j, :) = embeddingMatrix(seq(i, j) + 1, :);
    end
end
end    