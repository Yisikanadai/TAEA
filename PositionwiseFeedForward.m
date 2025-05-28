function output = PositionwiseFeedForward(x,params)
W1 = params.encoder.W1;
W2 = params.encoder.W2;
b1 = params.encoder.b1;
b2 = params.encoder.b2;

x = permute(x, [2, 3, 1]);
x = activationFunction(pagemtimes(W1 , 'none',x,'transpose') + b1);
x = pagemtimes(W2 ,x)+ b2;
output  = permute(x,[3, 2, 1]);
end