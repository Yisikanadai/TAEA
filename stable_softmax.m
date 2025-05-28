function y = stable_softmax(x)
x_max = max(x, [], 3);
x_max = reshape(x_max, [size(x, 1), size(x, 2), 1]);
x_exp = exp(x - x_max);
sum_exp = sum(x_exp, 3);
sum_exp = reshape(sum_exp, [size(x, 1), size(x, 2), 1]);
sum_exp(sum_exp == 0) = eps;
y = x_exp ./ sum_exp;
end