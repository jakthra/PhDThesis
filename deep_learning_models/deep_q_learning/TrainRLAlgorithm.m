clear all 
close all
SINR = 5;
interferingSubcarriers = 1:300;
episodes = 500;
RewardPenality = -1;

% Setup environment
env = MonsterEnv(SINR, interferingSubcarriers, true, RewardPenality);
BinaryReward = env.BinaryReward;

% Setup network
criticNetwork = createCriticNetwork();

% Setup training options
agent = setupAgent(criticNetwork, env);

% Train RL algorithm
[trainedAgent, trainingStats] = trainAgent(agent, env, episodes);
 

temp =  java.util.UUID.randomUUID;
myuuid = temp.toString;
folder = sprintf('Results/Trained_Agents/%s', myuuid);
if ~exist(folder,'dir')
	mkdir(folder)
end

save(sprintf('%s/config.mat',folder),'SINR', 'interferingSubcarriers','BinaryReward','RewardPenality')
save(sprintf('%s/agent.mat', folder),'trainedAgent')
save(sprintf('%s/network.mat', folder),'criticNetwork')
save(sprintf('%s/trainingStats.mat', folder),'trainingStats')

fprintf('Saved to: %s \n', folder)


