classdef MyEnv < rl.env.MATLABEnvironment

    properties
        data = []
        info = []
        PopulationSize = 40
        IndividualLength
        NumOperators = 20
        StateSize = 4
        ActionSize
        CurrentPopulation
        CurrentState
        fitnew
        fitold
        StepCount = 0
    end

    methods
        function this = MyEnv(data, info)
            ObservationInfo = rlNumericSpec([1 4]);
            ObservationInfo.Name = 'Population State';
            ObservationInfo.Description = 'Standard Deviation and Mean of Fitness';
            ActionInfo = rlFiniteSetSpec({1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20});
            ActionInfo.Name = 'Operator Action';
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            this.data = data;
            this.info = info;
            this.IndividualLength = data.n * 3;
            this.ActionSize = this.NumOperators;
            this.CurrentPopulation = rand(info.np, data.n*3);
            this.fitold = decode_01(this.CurrentPopulation, info, data);
            this.fitnew = this.fitold;
            [inputFeatures, ~] = calculate_input(this.CurrentPopulation, this.info, this.data, this.fitnew, this.fitold);
            this.CurrentState = round(inputFeatures);
            this.StepCount = 0;
        end

        function [Observation, Reward, IsDone, Info] = step(this, Action)
            this.StepCount = this.StepCount + 1;
            [~, ~, oldfit] = calculate_std(this.CurrentPopulation, this.info, this.data);
            [this.info, newPopulation, newfit] = RL_action(this.CurrentPopulation, this.data, this.info, Action, oldfit);
            this.fitnew = newfit;
            [inputFeatures, ~] = calculate_input(newPopulation, this.info, this.data, newfit, oldfit);
            newState = inputFeatures;
            Reward = RL_reward(newfit, oldfit, this.info, this.StepCount);
            this.CurrentPopulation = newPopulation;
            this.CurrentState = newState;
            Observation = this.CurrentState;
            IsDone = false;
            Info = [];
        end

        function InitialObservation = reset(this)
            this.CurrentPopulation = rand(this.info.np, this.data.n*3);
            this.fitold = decode_01(this.CurrentPopulation, this.info, this.data);
            this.fitnew = this.fitold;
            this.StepCount = 0;
            [inputFeatures, ~] = calculate_input(this.CurrentPopulation, this.info, this.data, this.fitnew, this.fitold);
            InitialObservation = this.CurrentState;
        end
    end
end