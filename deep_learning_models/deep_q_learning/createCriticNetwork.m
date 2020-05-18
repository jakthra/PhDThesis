function [criticNetwork] = createCriticNetwork()

observationPath = [
	imageInputLayer([600 140 1],"Name","observation", 'Normalization','none')
	
	convolution2dLayer([5 5],10,"Name","conv_1","Padding","same")
	dropoutLayer(0.01,"Name","dropout_1")
	%batchNormalizationLayer("Name","batch_norm1")
	reluLayer("Name","relu_1")
	maxPooling2dLayer(2,'Stride',2,"Name","pool_1")
	
	convolution2dLayer([2 2],10,"Name","conv_2","Padding","same","Stride",[1 1])
	dropoutLayer(0.05,"Name","dropout_2")
	%batchNormalizationLayer("Name","batch_norm2")
	reluLayer("Name","relu_2")
	maxPooling2dLayer(2,'Stride',2,"Name","pool_2")
	
	convolution2dLayer([2 2],10,"Name","conv_3","Padding","same","Stride",[1 1])
	dropoutLayer(0.05,"Name","dropout_3")
	%batchNormalizationLayer("Name","batch_norm3")
	reluLayer("Name","relu_3")
	maxPooling2dLayer(2,'Stride',2,"Name","pool_3")
	
	fullyConnectedLayer(10,"Name","CriticObsFC1")
	dropoutLayer(0.1,"Name","CriticObsFC1_dropout")];
	
	
actionPath = [
    imageInputLayer([1 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(10,'Name','CriticActFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','output')];
criticNetwork = layerGraph(observationPath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);    
criticNetwork = connectLayers(criticNetwork,'CriticObsFC1_dropout','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActFC1','add/in2');

end

