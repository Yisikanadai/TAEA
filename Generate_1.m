function processed_data = Generate_1(all_data, data, num_samples)
    processed_data = cell(num_samples, 1);
    for i = 1:num_samples
        temp = rand;
        if temp < 0.5
            temp2 = randi([1 ceil(data.n/2)-1]);
            temp3 = randi([ceil(data.n/2)+1 data.n]);
            processed_data{i} = all_data(i, temp2:temp3);
        else
            processed_data{i} = all_data(i, :);
        end
    end
end