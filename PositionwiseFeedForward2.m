function output = PositionwiseFeedForward2(x, params)
W1 = params.enc_dec.W1;
W2 = params.enc_dec.W2;
b1 = params.enc_dec.b1;
b2 = params.enc_dec.b2;
x = permute(x, [2, 3, 1]);
x = relu(pagemtimes(W1, 'none', x, 'transpose') + b1);
x = pagemtimes(W2, x) + b2;
output = permute(x, [3, 2, 1]);
end