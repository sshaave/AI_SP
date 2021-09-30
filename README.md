# AI_SP
AI_SP


Basic functioning AI consisting of two dense layers of neurons. ReLU and Softmax activation functions with different optimizers; where ADAM is the most reliable one for this problem. 

Goal:
Return the steel profile which yields the lowest GWP (global warming potential).
User is required to give the two following inputs: span length [m] and distributed load [kN/m]. 

As of now the AI has been trained to return a IPE, HEA or a SHS. 


The following is considered in the AIs scheme: 
- moment capasity
- shear capasity
- deformation

Limitations: 
- pointloads not considered
- only evenly distributed loads
- buckling not considered
- warping not considered
- NNFS pacakge required (used for replicating outputs from numpy randn()
