# N-Gram
This is an implementation in Unity for N-Gram prediction.

Article can be found in
[Implementing N-Grams for Player Prediction, Procedural Generation, and Stylized AI](http://www.gameaipro.com/GameAIPro/GameAIPro_Chapter48_Implementing_N-Grams_for_Player_Prediction_Proceedural_Generation_and_Stylized_AI.pdf) by Joseph Vasquez II

Using Job System to support muti-thread inference.

## Algorithm

A pseudocode algorithm for finding which N value provides the highest
prediction accuracy for pre-existing event sequences.
```
Instantiate an N-gram for each candidate N value ranged [1,20]
Create accuracy result storage for each N-gram
For each event sequence,
    For each event e ∈[0,L-1] in the sequence,
        For each N-gram ng,
            store result of e compared to ng.Predict()
            ng.UpdateWithNewEvent(e)
output accuracy results
output N value of most accurate N-gram based on results
```