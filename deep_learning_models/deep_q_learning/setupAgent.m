function [agent] = setupAgent(criticNetwork, env)


opt = rlRepresentationOptions('LearnRate',1e-4,'UseDevice',"cpu");
actInfo = getActionInfo(env);
actDimensions = actInfo.Dimension;
obsInfo = getObservationInfo(env);
obsDimensions = obsInfo.Dimension;

critic = rlRepresentation(criticNetwork,'Observation',{'observation'},obsInfo,'Action',{'action'},actInfo,opt);
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false,...    
    'TargetUpdateMethod',"smoothing",...
    'TargetSmoothFactor',1e-2,... 
    'ExperienceBufferLength',2000,... 
    'DiscountFactor',0.7,...
    'MiniBatchSize',32);
agentOpts.EpsilonGreedyExploration.Epsilon = 0.3;
agent = rlDQNAgent(critic,agentOpts);
end

