function lr = noam_scheme(global_step, warmup_steps,total_steps)
lr_max = 0.01;
lr_min = 0.0001;
step = double(global_step);
lr_noam = lr_max*global_step/warmup_steps;
cosine_lr = lr_min+(lr_max-lr_min)* 0.5 * (1 + cos(pi * (step - warmup_steps) / (total_steps - warmup_steps)));
if step <= warmup_steps
    lr = lr_noam;
else
    lr = cosine_lr;
end
lr = dlarray(lr);
end