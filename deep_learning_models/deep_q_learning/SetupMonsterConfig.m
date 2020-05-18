function Config = SetupMonsterConfig(totalRounds, numUsers)

    Config = MonsterConfig();
		Config.Runtime.simulationRounds = totalRounds;
    Config.Scenario = 'Single Cell';
    Config.MacroEnb.ISD = 300;
    Config.MacroEnb.sitesNumber = 1;
    Config.MacroEnb.cellsPerSite = 1;
    Config.MacroEnb.height = 35;
    Config.MicroEnb.sitesNumber = 0;
    Config.Ue.number = numUsers;
    Config.Ue.height = 1.5;
		Config.Ue.numPRBs = 50;
    Config.Traffic.primary = 'fullBuffer';
		Config.Traffic.arrivalDistribution = 'Static';
    Config.Traffic.mix = 0;
    Config.SimulationPlot.runtimePlot = 0;
		Config.Channel.interferenceType = 'Frequency';
		Config.Channel.fadingActive = 1;
		Config.Channel.shadowingActive = 0;

		Config.Phy.uplinkFrequency = 2000;


end