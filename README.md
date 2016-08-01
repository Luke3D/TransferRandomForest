# Transfer learning for Random Forest
Implementation of the SER-STRUCT algorithm to perform Transfer Learning with Random Forest

The algorithm is presented in this paper:

https://arxiv.org/pdf/1511.01258v2.pdf

Algorithm
-------------
The general idea of transfer learning is to learn a model on a *Source* (S) domain, where there is abundance of data, and refine it on a related *Target* (T) domain, where data availability is scarce and expensive. Therefore, a general assumption is that the amount of data D in the Source domain D(S) is >> D(T). 

We consider the scenario of learning a classifier (Random Forest) for activity recognition using a waist-worn wearable sensor: we have data from healthy subjects performing 5 type of physical activities (Sit, Stand, Stairs Up, Stairs Dw, Walking) and we train a model (classifier) that predicts an activity from the sensor data; this representes our Source domain. Now we want to refine our model to predict activities in patients with lower-limb impairments (Target domain), whose gait patterns are altered from that of healthy. Collecting data from patients is expensive, therefore we want to use the data previously collected in healthy (which is abundant) and only use little data from the patients. Therefore, we pose the problem as one of transfer learning.  
Another example (as described in the above paper) could be that of credit scoring: we have a model (classifier) trained on data from bank A (Source), and we ask whether we can refine the rules learned for A to perform credit scoring for a new bank B (Target).


Installation
-------------
Work in Progress - The code uses scikit-learn functions for making the random forest. 

Results
-------------
Work in Progress
