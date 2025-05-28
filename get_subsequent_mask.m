function mask = get_subsequent_mask(tra_inp, pad_idx)
[sz_b, len_s] = size(tra_inp);
mask1 = get_pad_mask(tra_inp, pad_idx);
mask1 = reshape(mask1, [sz_b, 1, len_s]);
subsequent_mask = triu(ones(len_s, len_s), 1);
subsequent_mask = reshape(subsequent_mask, [1, len_s, len_s]);
mask = mask1 | subsequent_mask;
end