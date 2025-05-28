function output = dropout1(input, dropout_rate)
mask = rand(size(input)) > dropout_rate;
output = input .* mask;
output = output / (1 - dropout_rate);
end