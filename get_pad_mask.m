function mask = get_pad_mask(seq, pad_idx)
mask = logical(seq == pad_idx);
end