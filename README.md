# Transfer learning for Random Forest
Implementation of the SER-STRUCT algorithm to perform Transfer Learning with Random Forest

The algorithm is presented in this paper:

http://ieeexplore.ieee.org/document/7592407/

Algorithm
-------------
The general idea of transfer learning is to learn a model on a *Source* (S) domain, where there is abundance of data, and refine it on a related *Target* (T) domain, assumed to be a variation of the source, and where data availability is scarce and/or expensive. Therefore, a general assumption is that the amount of data D in the Source domain D(S) is >> D(T). An example (as described in the above paper) is that of credit scoring: we have a model (classifier) trained on data from bank A (Source), and we ask whether we can refine the rules learned for A to perform credit scoring for a new bank B (Target).


We consider the scenario of learning a classifier (Random Forest) for activity recognition using a waist-worn activity monitor (accelerometer): we have data from healthy subjects performing 5 type of physical activities (Sit, Stand, Stairs Up, Stairs Dw, Walking) and we train a model (classifier) that predicts an activity from the sensor data; this represents our Source domain. Now we want to refine our model to predict activities in patients with lower-limb impairments (Target domain), whose gait patterns are altered from that of healthy. Collecting data from patients is expensive, therefore we want to use the data previously collected in healthy (which is abundant) and only use little data from the patients. Therefore, we pose the problem as one of transfer learning.  


## Installation

Clone the repository
```bash
  $ git clone https://github.com/Luke3D/TransferRandomForest.git
```
The example notebook shows the application of the algorithm to the Activity recognition scenario described above.
## Dependencies
 - [numpy](http://www.numpy.org/)
 - [matplotlib](http://matplotlib.org/)
 - [scikit-learn](http://scikit-learn.org/stable/install.html)
 - [Graphlab Create](https://turi.com/download/install-graphlab-create.html)

## Members
- [Luca Lonini](http://kordinglab.com/people/luca_lonini/)
- [Roozbeh Farhoodi](http://kordinglab.com/people/roozbeh_farhoodi/index.html)

## Acknowledgements
 - [Konrad Koerding](http://kordinglab.com) and [Arun Jayaraman](http://rto.smpp.northwestern.edu/) for support
 - [COGC](http://cogc.ir/) and [Ottobock](http://www.ottobockus.com/) for funding
 - Package is developed in [Konrad Kording's Lab](http://kordinglab.com/) at Northwestern University

## License
MIT License Copyright (c) 2016 Luca Lonini and Roozbeh Farhoodi
