function [batchEncoderInput, batchDecoderInput, batchDecoderTarget] = prepareInputsTargets(batchACOSeq, startToken, endToken, padToken, vocabSize)
maxLen = max(cellfun(@length, batchACOSeq));
batchEncoderInput = cell2mat(cellfun(@(x) padSequence([startToken, x, endToken], maxLen + 2, padToken), batchACOSeq, 'UniformOutput', false));
batchDecoderInput = cell2mat(cellfun(@(x) padSequence([startToken, x], maxLen + 1, padToken), batchACOSeq, 'UniformOutput', false));
batchDecoderTarget = cell2mat(cellfun(@(x) padSequence([x, endToken], maxLen + 1, padToken), batchACOSeq, 'UniformOutput', false));
end    