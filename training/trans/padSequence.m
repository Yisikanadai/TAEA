function paddedSeq = padSequence(seq, maxLen, padToken)
paddedSeq = [seq, repmat(padToken, 1, maxLen - length(seq))];
end    