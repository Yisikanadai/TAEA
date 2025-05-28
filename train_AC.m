clear
clc;
load datain_TAEA.mat

info.mode = 1;
info.num = 1;
prefix = 'fft';
if info.mode == 2
    prefix = 'gs';
end
field = [prefix, num2str(info.num)];
data = datain_TAEA.(field);
info.flag = 7;

info = setuppara(info);
[info, sol, ~] = heft_sol(info, data);
info.fitbest = decode(sol, info, data);
info.solbest = sol;

env = MyEnv(data, info);
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
validateEnvironment(env);
[~, ~, agent] = build_AC(obsInfo, actInfo);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 50, ...
    'MaxStepsPerEpisode', 500, ...
    'StopTrainingCriteria', 'EpisodeReward', ...
    'StopTrainingValue', 10000, ...
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 1000, ...
    'Verbose', true, ...
    'Plots', 'none' ...
);
train(agent, env, trainOpts);
save('AC_model.mat', 'agent', 'trainOpts');