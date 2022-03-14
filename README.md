# ar-border-predictor

This library implements a way to find a border for a stochastic process 
(approximated as AR model) at a moment of time in the future such that
at least one of the previous further values is less than the border with the 
given probability.

## Usage example

```javascript
const arbp = require('ar-border-predictor')

// Time series
const z = [1, 2, 1.1, 2, 1, 2]

// Create a model object (dimension is 1)
const model = new arbp.ARModel(1)

// Train the model
model.fit(z)

// Predict next 10 values randomly via the model
const h = model.simulate(z, 10)
console.log(h)

// Create border predictor instance with 100 simulations
const bp = new arbp.BorderPredictor(100)

// Simulate next 100 values for each trajectory with the trained model
bp.simulate(z, 100, model)

// Predict the border with the probability 0.9 on the step 50
const border = bp.predict(0.9, 50)
console.log(border)
````
