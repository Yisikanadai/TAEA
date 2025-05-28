function output = MultiHeadAttention(embeddedInputData, pad_idx, d_k, d_v, numHeads, params, encoderInput)
[batch_size, seq_len1, ~] = size(embeddedInputData);
seq_len2 = seq_len1;
seq_len3 = seq_len1;
embeddedInputData = permute(embeddedInputData, [2, 3, 1]);

q = pagemtimes(embeddedInputData, params.encoder.Wq);
k = pagemtimes(embeddedInputData, params.encoder.Wk);
v = pagemtimes(embeddedInputData, params.encoder.Wv);

q = permute(q, [3 1 2]);
k = permute(k, [3 1 2]);
v = permute(v, [3 1 2]);

q = reshape(q, [batch_size, seq_len1, numHeads, d_k]);
k = reshape(k, [batch_size, seq_len2, numHeads, d_k]);
v = reshape(v, [batch_size, seq_len3, numHeads, d_v]);

q = permute(q, [2 4 3 1]);
k = permute(k, [2 4 3 1]);

q = reshape(q, [seq_len1, d_k, batch_size * numHeads]);
k = reshape(k, [seq_len2, d_k, batch_size * numHeads]);

mask = get_pad_mask(encoderInput, pad_idx); 

mask = reshape(mask, [batch_size, 1, seq_len2]);
mask = repmat(mask, [numHeads, 1, 1]); 

attention_scores = pagemtimes(q, 'none', k, 'transpose') / sqrt(d_k);
attention_scores = permute(attention_scores, [3 1 2]);
attention_scores = attention_scores + (mask * -1e9);

attention_probs = stable_softmax(attention_scores);
attention_probs = permute(attention_probs, [1 3 2]);
attention_probs = permute(attention_probs, [3 2 1]);

v = permute(v, [2 4 3 1]);
v = reshape(v, [seq_len3, d_v, batch_size * numHeads]);
Oattention = pagemtimes(attention_probs, v); 
Oattention = permute(Oattention, [3 1 2]);
Oattention = reshape(Oattention, [batch_size, numHeads, seq_len1, d_v]);
Oattention = permute(Oattention, [1 3 2 4]);
Oattention = reshape(Oattention, [batch_size, seq_len1, numHeads * d_v]);
Oattention = permute(Oattention, [2 3 1]);
output = pagemtimes(Oattention, params.encoder.Wo);
end