function [fit,able,pop_true,mt_min] = decode_g1(pop,info,data)
pop_size = size(pop,1);
fit = zeros(1,pop_size);
pop_true = zeros(info.np, data.n*3);
able = zeros(1,pop_size);
for i = 1 : pop_size
    pop(i,1:data.n) = ceil(pop(i,1:data.n) * data.m);
    [~,pop(i,data.n+1:data.n*2)] = sort(pop(i,data.n+1:data.n*2) * data.n);
end
mt_min=10000;
for k = 1 : pop_size
    [fit(k),able(k),~,pop_true(k,:),mt_max] = decode_g2(pop(k,:),info,data);
    mt_min=min(mt_max,mt_min);
end

end