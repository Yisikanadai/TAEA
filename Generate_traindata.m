clear;
clc;
load datain_TAEA.mat

num_len=200;
info.mode = 1;
info.num = 1;

prefix = 'fft';
if info.mode == 2
    prefix = 'gs';
end

field = [prefix, num2str(info.num)];
data = datain_TAEA.(field);

info = setuppara_g(info,data);
pophis = [];

for N = 1 : 5
    pophis_temp = zeros(200, info.n*3);
    idx = 1;
    
    pop = rand(info.np, info.n*3);
    fit_next = zeros(1,info.np);
    mt_min=10000;
    
    [fit,~,~] = decode_g1(pop,info,data);
    m = 0;
    
    tic
    for t = 1 : info.ng * 3
        time = toc;
        if time > info.maxrt
            break;
        end
        
        [~,index] = min(fit);
        pop_min = pop(index,:);
        
        popnext = Generate(pop, info, fit);
        [fit_next, able, pop_true, mt_now] = decode_g1(popnext,info,data);
        mt_min=min(mt_now);
        
        pop_temp = pop;
        for i = 1: info.np
            if able(i) == 1
                pop_temp(i, data.n+1:data.n*2) = pop_true(i, data.n+1:data.n*2);
                pophis_temp(idx,:) = pop_temp(i,:);
                idx = idx + 1;
                if idx >= 200+1
                    idx = 1;
                end
            end
        end
        
        for i=1:info.np
            if fit_next(i)<fit(i)
                pop(i,:)=popnext(i,:);
                fit(i)=fit_next(i);
            end
        end
        
        temp = ['Iteration:', num2str(t), ',Best Fit:',num2str(min(fit))];
        disp(temp);
    end
    
    pophis_temp = pophis_temp(~all(pophis_temp == 0, 2), :);
    len_temp = size(pophis_temp, 1);
    fprintf('len_temp: %d\n', len_temp);
    
    num_pophis = num_len;
    if len_temp >= num_pophis
        pophis = [pophis; pophis_temp(len_temp-num_pophis+1: len_temp,:)];
    elseif len_temp < num_pophis
        start_idx = max(ceil(len_temp / 2)-1, 1);
        pophis = [pophis; pophis_temp(start_idx : len_temp, :)];
    end
end

all_data = pophis;
num_samples = size(pophis, 1);

train_data = Generate_1(all_data, data, num_samples);

aco_data = zeros(num_samples, data.n);
seq_data = zeros(num_samples, data.n);
eng_train = pophis(:, data.n*2+1:end);

for i = 1:num_samples
    aco_data(i,:) = ceil(pophis(i,1:data.n) * data.m);
    seq_data(i,:) = pophis(i, data.n+1:2*data.n);
end

aco_train = Generate_1(aco_data, data, num_samples);
seq_train = Generate_1(seq_data, data, num_samples);

save('train.mat', 'seq_train', 'aco_train', 'train_data', 'data', 'info');