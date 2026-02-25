
Codes for generating the results for the paper : 
**"Filling Causal Responsibility Gaps in Spatial Interactions using Feasible Action-Space Reduction by Groups"**

## Contributors

 *Ashwin George*
 - github.com/ashwin-geo
 - A.George@tudelft.nl

*Vassil Guenov* 
- github.com/vguenov
- v.v.guenov@student.tudelft.nl

## Instructions

- Run `GroupRespSim.ipynb` for running simulations, computing group FeARs and the tiering algorithm.
- Run `GroupResp_Illustrations.ipynb` to generate illustrations in the paper.
- Run `ReadAndPlot-GroupRanks.ipynb` for reading and plotting the results of the randomised simulations.

## Information

- Read https://github.com/DAI-Lab-HERALD/FeAR/blob/main/README.md for an overview of the **Grid World environment** and how FeAR can be computed
- Function for computing group FeAR can be found in `Responsibility.py`
- The tiering algorithm can be found in `GroupSort.py`
- Results for gFeAR are stored in `gFeAR_Results`
	- We have already included the results for the randomised simulations used in the paper.
- To build new scenarios use `ScenarioBuilder.ipynb` 
