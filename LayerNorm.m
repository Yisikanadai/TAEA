function output = LayerNorm(x, params)
epsilon = 1e-6;
mu = mean(x, 3);
sigma = std(x, 0, 3);
output = (x - mu) ./ (sigma + epsilon);
output = output .* params.gamma + params.beta;
end