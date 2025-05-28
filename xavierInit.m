function W = xavierInit(n_in, n_out)
limit = sqrt(6 / (n_in + n_out));
W  = rand(n_out, n_in) * 2 * limit - limit;
end